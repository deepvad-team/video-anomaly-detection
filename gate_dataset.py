import numpy as np
import torch
import torch.utils.data as data


def normalize_video_feature_shape(x_video_np):
    """
    입력 feature를 (T, D)로 맞춤
    가능한 입력:
      (T, D)
      (T, C, D)  -> crop 평균
      (T, 1, C, D) 등 -> squeeze 후 crop 평균
    """
    x = np.asarray(x_video_np)

    if x.ndim == 2:
        return x

    if x.ndim == 3:
        # (T, C, D)
        return x.mean(axis=1)

    if x.ndim == 4:
        x = np.squeeze(x)
        if x.ndim == 2:
            return x
        elif x.ndim == 3:
            return x.mean(axis=1)

    raise ValueError(f"Unsupported feature shape: {x.shape}")


class PrefixGateVideoDataset(data.Dataset):
    """
    flat concat feature + pseudo label + nalist를 이용해
    video 단위 episode를 반환

    반환:
      x_video: (T, D)
      y_video: (T,)
      vid_idx: int
    """
    def __init__(
        self,
        conall_path,
        pseudo_path,
        nalist_path,
        dtype="float32",
    ):
        self.nalist = np.load(nalist_path)
        self.pseudo = np.load(pseudo_path).astype(np.float32)

        total_T = int(self.nalist[-1, 1])

        # 먼저 memmap을 3D로 열어봄: (total_T, 10, 2048)
        # 안 맞으면 fallback으로 2D 시도 가능
        try:
            self.con_all = np.memmap(
                conall_path,
                dtype=dtype,
                mode="r",
                shape=(total_T, 10, 2048)
            )
            self.feature_mode = "crop10"
        except Exception:
            # fallback
            self.con_all = np.memmap(
                conall_path,
                dtype=dtype,
                mode="r",
                shape=(total_T, 2048)
            )
            self.feature_mode = "flat2048"

        assert len(self.pseudo) == total_T, \
            f"Pseudo label length mismatch: {len(self.pseudo)} vs total_T={total_T}"

        print(f"[PrefixGateVideoDataset] con_all mode={self.feature_mode}, shape={self.con_all.shape}")
        print(f"[PrefixGateVideoDataset] pseudo shape={self.pseudo.shape}")
        print(f"[PrefixGateVideoDataset] nalist shape={self.nalist.shape}")

    def __len__(self):
        return len(self.nalist)

    def __getitem__(self, vid_idx):
        s, e = self.nalist[vid_idx]
        s, e = int(s), int(e)

        x_video_np = self.con_all[s:e]     # (T, 10, 2048) or (T, 2048)
        x_video_np = normalize_video_feature_shape(x_video_np)  # -> (T, 2048)

        y_video_np = self.pseudo[s:e]      # (T,)

        x_video = torch.from_numpy(np.asarray(x_video_np, dtype=np.float32))
        y_video = torch.from_numpy(np.asarray(y_video_np, dtype=np.float32))

        return x_video, y_video, vid_idx