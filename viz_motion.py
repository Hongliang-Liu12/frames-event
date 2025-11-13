import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import dataset and helpers from frames_eval (safe because main guard prevents execution)
from frames_eval import Evdet200kCocoDataset, letterbox_collate_fn
from nets.yolo_frames_net import YoloBodySST
from loguru import logger


CHECKPOINT_PATH = "/home/lhl/Git/frames-event/logs/newtwostage/only2_perfect_resume/42_7_0.0002_0.00015_4_model_B_EMA_ep010_map-0.5260.pth"
DATASET_ROOT_DIR = "/home/lhl/Git/datasets/EvDET200K"
OUT_DIR = "viz_outputs"
BATCH_SIZE = 1
SEQ_LEN = 3
CONFIDENCE = 0.01


def to_uint8_map(x):
    """Normalize a float numpy array to uint8 0..255"""
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    x = (x * 255.0).astype(np.uint8)
    return x


def save_heatmap(map_hw, outpath, colormap=cv2.COLORMAP_JET):
    m8 = to_uint8_map(map_hw)
    colored = cv2.applyColorMap(m8, colormap)
    cv2.imwrite(outpath, colored)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading dataset (test)...")
    dataset = Evdet200kCocoDataset(DATASET_ROOT_DIR, split="test", seq_len=SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False, collate_fn=letterbox_collate_fn)

    logger.info(f"Loading model from {CHECKPOINT_PATH}...")
    model = YoloBodySST(num_classes=len(dataset.coco.getCatIds()), phi='s', num_frame=SEQ_LEN)
    model = model.to(device)

    # Robust loading similar to frames_eval
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    except ModuleNotFoundError as e:
        msg = str(e)
        if 'mmengine' in msg:
            import pickle, io, types

            class SafeUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module.startswith('mmengine'):
                        return lambda *a, **k: None
                    return super().find_class(module, name)

            def _safe_loads(b):
                return SafeUnpickler(io.BytesIO(b)).load()

            fake_pickle = types.SimpleNamespace(loads=_safe_loads)
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, pickle_module=fake_pickle)
        else:
            raise

    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
        elif 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']

    # Filter matching keys
    model_state = model.state_dict()
    load_dict = {k: v for k, v in checkpoint.items() if k in model_state and model_state[k].shape == v.shape}
    model_state.update(load_dict)
    model.load_state_dict(model_state)
    logger.info("Model loaded.")

    # Prepare hooks to capture intermediate tensors
    captured = {}

    def make_hook(name):
        def hook(module, input, output):
            # Save CPU detached clone
            try:
                captured[name] = output.detach().cpu()
            except Exception:
                # Some modules may return tuples
                captured[name] = output
        return hook

    # Register hooks on motion/context/gate/final inside EnhancedTemporalNeck for c3/c4/c5
    model.neck_c3.conv_motion.register_forward_hook(make_hook('c3_motion'))
    model.neck_c3.conv_context.register_forward_hook(make_hook('c3_context'))
    model.neck_c3.gate.register_forward_hook(make_hook('c3_gate'))
    model.neck_c3.conv_final.register_forward_hook(make_hook('c3_final'))

    model.neck_c4.conv_motion.register_forward_hook(make_hook('c4_motion'))
    model.neck_c4.conv_context.register_forward_hook(make_hook('c4_context'))
    model.neck_c4.gate.register_forward_hook(make_hook('c4_gate'))
    model.neck_c4.conv_final.register_forward_hook(make_hook('c4_final'))

    model.neck_c5.conv_motion.register_forward_hook(make_hook('c5_motion'))
    model.neck_c5.conv_context.register_forward_hook(make_hook('c5_context'))
    model.neck_c5.gate.register_forward_hook(make_hook('c5_gate'))
    model.neck_c5.conv_final.register_forward_hook(make_hook('c5_final'))

    # attention maps
    model.attention_map_c3.register_forward_hook(make_hook('attn_c3'))
    model.attention_map_c4.register_forward_hook(make_hook('attn_c4'))
    model.attention_map_c5.register_forward_hook(make_hook('attn_c5'))

    # Capture the final fused_c* inputs passed into FPN by hooking fpn (it receives tuple(final_c3, final_c4, final_c5))
    def fpn_hook(module, input, output):
        # input is a tuple with a single argument: the tuple of final_c3,c4,c5
        try:
            finals = input[0]
            captured['finals_pre_fpn'] = tuple([t.detach().cpu() for t in finals])
        except Exception:
            captured['finals_pre_fpn'] = input
        # also save outputs
        try:
            captured['fpn_outputs'] = tuple([t.detach().cpu() for t in output])
        except Exception:
            captured['fpn_outputs'] = output

    model.fpn.register_forward_hook(fpn_hook)

    # Run a single batch
    model.eval()
    with torch.no_grad():
        for images_batch, targets, img_infos, ratios, pads in dataloader:
            images = images_batch.to(device)
            # forward
            outputs = model(images)

            # For visualization we will use the first sample in the batch
            info = img_infos[0]
            ratio = ratios[0]
            pad_left, pad_top = pads[0]

            # Read original image (last frame in the sequence)
            # The dataset stores images under dataset.root + file_name
            file_name = info.get('file_name')
            img_path = os.path.join(dataset.root, file_name)
            orig_img = cv2.imread(img_path)
            if orig_img is None:
                # Fallback: use the padded tensor from images_batch
                padded = images_batch[0, -1].cpu().numpy().transpose(1,2,0).astype(np.uint8)
                orig_img = padded.copy()

            h, w = info.get('height', orig_img.shape[0]), info.get('width', orig_img.shape[1])

            # Save original last frame
            cv2.imwrite(os.path.join(OUT_DIR, 'original_last_frame.png'), orig_img)

            # Save the original input sequence frames (unpadded, full resolution) using dataset mapping
            seq_filenames = []
            try:
                # target image id (the key frame)
                target_img_id = info.get('id')
                seq_ids = dataset.id_to_sequence.get(target_img_id, None)
                if seq_ids is None:
                    # fallback: try to use the images in the preprocessed batch
                    raise KeyError('sequence ids not found')

                for idx, frame_id in enumerate(seq_ids):
                    frame_info = dataset.original_coco_loader.loadImgs(frame_id)[0]
                    frame_name = frame_info.get('file_name')
                    frame_path = os.path.join(dataset.root, frame_name)
                    frame_img = cv2.imread(frame_path)
                    if frame_img is None:
                        logger.warning(f"Could not read original frame {frame_path}, using black image as fallback")
                        frame_img = np.zeros((h, w, 3), dtype=np.uint8)
                    else:
                        # If the original image shape differs, resize to match single target size for grid
                        frame_img = cv2.resize(frame_img, (w, h), interpolation=cv2.INTER_LINEAR)

                    fname = os.path.join(OUT_DIR, f'seq_frame_{idx}.png')
                    cv2.imwrite(fname, frame_img)
                    seq_filenames.append(fname)
            except Exception:
                # Fallback: use the preprocessed frames (letterboxed) if original frames can't be loaded
                logger.warning('Falling back to preprocessed frames for sequence display')
                for i in range(images_batch.shape[1]):
                    frame = images_batch[0, i].cpu().numpy().transpose(1,2,0)
                    frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
                    # remove center padding then resize back to original image size
                    new_w = int(w * ratio)
                    new_h = int(h * ratio)
                    # guard crop coords
                    x0 = int(pad_left)
                    y0 = int(pad_top)
                    x1 = x0 + new_w
                    y1 = y0 + new_h
                    crop = frame_uint8[y0:y1, x0:x1]
                    if crop.size == 0:
                        # fallback to the full padded frame if cropping failed
                        crop = frame_uint8
                    frame_resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
                    fname = os.path.join(OUT_DIR, f'seq_frame_{i}.png')
                    cv2.imwrite(fname, frame_resized)
                    seq_filenames.append(fname)

            # Helper to upsample map to original image size
            def ups(map_tensor):
                # map_tensor: numpy HxW (in model feature/map coords)
                # Steps: resize to model input (640x640), remove center pad, then resize to original image (w,h)
                input_w, input_h = 640, 640
                up640 = cv2.resize(map_tensor, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
                new_w = int(w * ratio)
                new_h = int(h * ratio)
                x0 = int(pad_left)
                y0 = int(pad_top)
                x1 = x0 + new_w
                y1 = y0 + new_h
                crop = up640[y0:y1, x0:x1]
                if crop.size == 0:
                    crop = up640
                out = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
                return out

            # Process captured tensors for c3/c4/c5
            for lvl in ['c3', 'c4', 'c5']:
                motion = captured.get(f'{lvl}_motion')
                context = captured.get(f'{lvl}_context')
                gate = captured.get(f'{lvl}_gate')
                final = captured.get(f'{lvl}_final')

                if motion is not None:
                    # motion shape [B, C, H, W]
                    m = motion[0].cpu().float().numpy()
                    m_mean = np.mean(m, axis=0)
                    m_map = ups(m_mean)
                    save_heatmap(m_map, os.path.join(OUT_DIR, f'{lvl}_motion_mean.png'))

                if context is not None:
                    c = context[0].cpu().float().numpy()
                    c_mean = np.mean(c, axis=0)
                    c_map = ups(c_mean)
                    save_heatmap(c_map, os.path.join(OUT_DIR, f'{lvl}_context_mean.png'))

                if gate is not None:
                    g = gate[0].cpu().float().numpy()
                    g_mean = np.mean(g, axis=0)
                    g_map = ups(g_mean)
                    save_heatmap(g_map, os.path.join(OUT_DIR, f'{lvl}_gate_mean.png'))

                if final is not None:
                    f = final[0].cpu().float().numpy()
                    f_mean = np.mean(f, axis=0)
                    f_map = ups(f_mean)
                    save_heatmap(f_map, os.path.join(OUT_DIR, f'{lvl}_final_mean.png'))

            # Attention maps (already single channel)
            attn_maps = {}
            for lvl in ['c3', 'c4', 'c5']:
                attn = captured.get(f'attn_{lvl}')
                if attn is not None:
                    a = attn[0].squeeze(0).cpu().numpy()
                    a_map = ups(a)
                    attn_maps[lvl] = a_map
                    # overlay on image (per-level)
                    heat = to_uint8_map(a_map)
                    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(orig_img, 0.6, heat_color, 0.4, 0)
                    cv2.imwrite(os.path.join(OUT_DIR, f'{lvl}_attention_overlay.png'), overlay)

            # Create a combined attention overlay (average of available levels) and a combined grid image
            if attn_maps:
                # compute average attention
                maps = list(attn_maps.values())
                avg_attn = np.mean(np.stack(maps, axis=0), axis=0)
                avg_heat = to_uint8_map(avg_attn)
                avg_color = cv2.applyColorMap(avg_heat, cv2.COLORMAP_JET)
                avg_overlay = cv2.addWeighted(orig_img, 0.6, avg_color, 0.4, 0)
                cv2.imwrite(os.path.join(OUT_DIR, 'attention_combined_overlay.png'), avg_overlay)

                # Build combined image: top row = three sequence frames, bottom row = avg_overlay
                # Ensure we have exactly SEQ_LEN frames (if fewer, pad with black)
                tops = []
                for i in range(SEQ_LEN):
                    if i < len(seq_filenames):
                        fimg = cv2.imread(seq_filenames[i])
                    else:
                        fimg = np.zeros_like(avg_overlay)
                    tops.append(fimg)
                top_row = cv2.hconcat(tops)
                bottom_row = cv2.resize(avg_overlay, (top_row.shape[1], top_row.shape[0] // SEQ_LEN), interpolation=cv2.INTER_LINEAR)
                # To make bottom row height match top_row/SEQ_LEN ratio, tile avg_overlay vertically to the same height as one frame
                bottom_row_full = cv2.resize(avg_overlay, (top_row.shape[1], tops[0].shape[0]), interpolation=cv2.INTER_LINEAR)
                combined = cv2.vconcat([top_row, bottom_row_full])
                cv2.imwrite(os.path.join(OUT_DIR, 'seq_then_attention_grid.png'), combined)

            # Save fusion gate scalars
            fg_c3 = model.fusion_gate_c3.item()
            fg_c4 = model.fusion_gate_c4.item()
            fg_c5 = model.fusion_gate_c5.item()
            with open(os.path.join(OUT_DIR, 'fusion_gate_values.txt'), 'w') as f:
                f.write(f'fusion_gate_c3: {fg_c3}\n')
                f.write(f'fusion_gate_c4: {fg_c4}\n')
                f.write(f'fusion_gate_c5: {fg_c5}\n')

            logger.info(f"Saved visualizations to {OUT_DIR}")
            break


if __name__ == '__main__':
    main()
