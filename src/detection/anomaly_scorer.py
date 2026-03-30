import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),   nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class AnomalyScorer:
    """Convolutional autoencoder-based anomaly scorer."""

    def __init__(self, latent_dim: int = 128, input_shape: Tuple = (64, 64, 3), threshold: float = 0.5):
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ConvAutoencoder(latent_dim).to(self.device)
        self.trained = False

    def _preprocess(self, frames: List[np.ndarray]) -> torch.Tensor:
        import cv2
        tensors = []
        h, w = self.input_shape[:2]
        for f in frames:
            if f.dtype != np.uint8:
                f = (f * 255).astype(np.uint8)
            f = cv2.resize(f, (w, h))
            t = torch.from_numpy(f).float().permute(2, 0, 1) / 255.0
            tensors.append(t)
        return torch.stack(tensors)

    def train(self, normal_frames: List[np.ndarray], epochs: int = 30, batch_size: int = 16) -> List[float]:
        data = self._preprocess(normal_frames).to(self.device)
        dataset = TensorDataset(data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        losses = []
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                recon = self.model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg = epoch_loss / len(loader)
            losses.append(avg)
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} — Loss: {avg:.4f}")
        self.trained = True
        self._calibrate_threshold(normal_frames)
        return losses

    def _calibrate_threshold(self, normal_frames: List[np.ndarray]):
        scores = [self._raw_score(f) for f in normal_frames[:100]]
        self.threshold = float(np.mean(scores) + 2 * np.std(scores))
        logger.info(f"Anomaly threshold set to {self.threshold:.4f}")

    def _raw_score(self, frame: np.ndarray) -> float:
        self.model.eval()
        with torch.no_grad():
            t = self._preprocess([frame]).to(self.device)
            recon = self.model(t)
            return float(nn.MSELoss()(recon, t).item())

    def score_frame(self, frame: np.ndarray) -> float:
        return self._raw_score(frame)

    def is_anomalous(self, frame: np.ndarray) -> Tuple[bool, float]:
        score = self.score_frame(frame)
        return score > self.threshold, score

    def score_sequence(self, frames: List[np.ndarray]) -> List[float]:
        return [self.score_frame(f) for f in frames]

    def set_threshold(self, threshold: float):
        self.threshold = threshold

    def save(self, path: str):
        torch.save({"model": self.model.state_dict(), "threshold": self.threshold}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.threshold = ckpt["threshold"]
        self.trained = True
