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
        for images_batch, targets, img_infos, ratios in dataloader:
            images = images_batch.to(device)
            # forward
            outputs = model(images)

            # For visualization we will use the first sample in the batch
            info = img_infos[0]
            ratio = ratios[0]

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

            # Save original
            cv2.imwrite(os.path.join(OUT_DIR, 'original_last_frame.png'), orig_img)

            # Helper to upsample map to original image size
            def ups(map_tensor):
                # map_tensor: numpy HxW
                return cv2.resize(map_tensor, (w, h), interpolation=cv2.INTER_LINEAR)

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
            for lvl in ['c3', 'c4', 'c5']:
                attn = captured.get(f'attn_{lvl}')
                if attn is not None:
                    a = attn[0].squeeze(0).cpu().numpy()
                    a_map = ups(a)
                    # overlay on image
                    heat = to_uint8_map(a_map)
                    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(orig_img, 0.6, heat_color, 0.4, 0)
                    cv2.imwrite(os.path.join(OUT_DIR, f'{lvl}_attention_overlay.png'), overlay)

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
