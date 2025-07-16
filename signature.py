#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────────────
#  Signature Generation & Verification Toolkit
#  • Synthetic-signature generator with rich distortion / noise pipeline
#  • Flexible Siamese network that can use a tiny CNN, ResNet-50 embedding, #    or a ViT-style transformer as its feature extractor
# ──────────────────────────────────────────────────────────────────────────────
#https://www.1001fonts.com/handwriting+messy-fonts.html?page=15
import os
import random,math
import string
from pathlib import Path
from typing import List, Tuple, Optional
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from joblib import Parallel, delayed
from IPython.display import clear_output
from sklearn.utils.validation import joblib
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from sklearn.model_selection import train_test_split
target_folder = "/content/drive/MyDrive/Projects/Signatures/"


def generate_random_string(length=np.random.randint(4, 16), use_digits=True, use_letters=True):
    if length is None:
        length = random.randint(5, 10)
    characters = ""
    if use_letters:
        characters += string.ascii_letters  # a–z + A–Z
    if use_digits:
        characters += string.digits         # 0–9

    if not characters:
        raise ValueError("At least one of use_digits or use_letters must be True.")

    return "".join(random.choice(characters) for _ in range(length))



# ═════════════════════════════════════════════════════════════════════════════
#  1.  Synthetic signature generator
# ═════════════════════════════════════════════════════════════════════════════
class Signature_Generator:
    """
    Creates printable PNG images that look like hand-written signatures, with optional geometric distortions and four noise types:
    gaussian | salt_pepper | scatter_high | none
    """

    # ────────────────────────────────────────────────────────────────────────
    #  Construction helpers
    # ────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _levenshtein_index(s1: str, s2: str) -> int:
        """Classic DP edit-distance but trimmed to similarity count only."""
        n1, n2 = len(s1), len(s2)
        n=np.abs(n1-n2)
        if n1>n2:
            s2=s2+" "*n
        else:
            s1=s1+" "*n
            
        dp = np.sum([s1[i]==s2[i] for i in range(n1)])
        return dp/len(s1)

    @staticmethod
    def _random_string(length: int = 10, use_digits: bool = True, use_letters: bool = True, ) -> str:
        chars = ""
        if use_letters:
            chars += string.ascii_letters
        if use_digits:
            chars += string.digits
        if not chars:
            raise ValueError("At least one of use_digits/use_letters must be True")
        return "".join(random.choice(chars) for _ in range(length))

    @staticmethod
    def _similar_strings(base: str, num_variants: int = 5, num_changes: int = 2, ) -> List[str]:
        variants = []
        for _ in range(num_variants):
            s = list(base)
            idx = random.sample(range(len(s)), min(num_changes, len(s)))
            for i in idx:
                s[i] = random.choice(string.ascii_letters + string.digits)
            variants.append("".join(s))
        variants=[base]+variants
        return variants

    # ────────────────────────────────────────────────────────────────────────
    #  InitialiserD:\Projects\SignatureProject\signature.py
    # ────────────────────────────────────────────────────────────────────────
    def __init__(self, name: str = "Rauf Giwa", 
                 target_folder: str = target_folder, 
                 signatures_folder: str = "Signatures/", 
                 font_folder: str = "Fonts/", dpi: int = 80, image_size: Tuple[int, int] = (400, 100), 
                 font_size: int = 80, text_color: str = "black", background_color: Tuple[int, int, int, int] = (255, 255, 255, 0), 
                 noise_type: str = "gaussian", distortion_factor: float = 1.5, scatter_radius: int = 15, scatter_points: int = 25,
                 threshold: int = 200, salt_vs_pepper: float = 0.1, salt_pepper_vol: float = 0.0005, noise_amount: int = 25, 
                 mean: float = 0.0, sigma: float = 25.0, ):
        self.name = name
        self.target_folder = target_folder
        self.signatures_folder = target_folder+signatures_folder
        self.font_folder = target_folder+font_folder
        print(self.font_folder)
        
        Path(self.signatures_folder).mkdir(parents=True, exist_ok=True)

        self.fonts = [f"{self.font_folder}{f}" for f in os.listdir(self.font_folder) if f.split(".")[-1].lower() in ("ttf", "otf")]
        if not self.fonts:
            raise RuntimeError("No fonts found in {}".format(font_folder))

        self.dpi = dpi
        self.image_size = image_size
        self.font_size = font_size
        self.text_color = text_color
        self.background_color = background_color

        self.noise_type = noise_type.lower()
        self.distortion_factor = distortion_factor
        self.scatter_radius = scatter_radius
        self.scatter_points = scatter_points
        self.threshold = threshold
        self.salt_vs_pepper = salt_vs_pepper
        self.salt_pepper_vol = salt_pepper_vol
        self.noise_amount = noise_amount
        self.mean = mean
        self.sigma = sigma
        self.pct_changes=[0.2,0.4,0.6,0.8,1.0]
        self.num_variants=5
        # interpolation options for visual inspection
        self.methods = [
            None, "none", "nearest", "bilinear", "bicubic", "spline16", "spline36", "hanning", "hamming", "hermite", "kaiser", "quadric", "catrom", "gaussian", "bessel", "mitchell", "sinc", "lanczos", ]

    # ────────────────────────────────────────────────────────────────────────
    #  Noise / distortion utilities
    # ────────────────────────────────────────────────────────────────────────
    def _scatter_high_values(self, img_np: np.ndarray) -> np.ndarray:
        """Randomly scatter the brightest pixels within a local radius."""
        scattered = img_np.copy()
        ys, xs = np.where(img_np > self.threshold)
        coords = list(zip(ys, xs))
        random.shuffle(coords)
        height, width = img_np.shape[:2]
        for (y, x) in coords[: self.scatter_points]:
            dy = random.randint(-self.scatter_radius, self.scatter_radius)
            dx = random.randint(-self.scatter_radius, self.scatter_radius)
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                scattered[ny, nx] = img_np[y, x]
        return scattered

    def _distort_high_values(self, img_np: np.ndarray) -> np.ndarray:
        """Multiply the brightest pixels by distortion_factor (clipped)."""
        distorted = img_np.copy()
        mask = distorted > self.threshold
        distorted[mask] = np.clip(distorted[mask] * self.distortion_factor, 0, 255
        )
        return distorted.astype(np.uint8)

    def _add_gaussian_noise(self, img_np: np.ndarray) -> np.ndarray:
        gauss = np.random.normal(self.mean, self.sigma, img_np.shape)
        noisy = np.clip(img_np.astype(np.int16) + gauss, 0, 255)
        return noisy.astype(np.uint8)

    def _add_salt_pepper_noise(self, img_np: np.ndarray) -> np.ndarray:
        noisy = img_np.copy()
        total = np.prod(img_np.shape[:2])
        num_salt = int(self.noise_amount * total * self.salt_vs_pepper)
        num_pepper = int(self.noise_amount * total * (1 - self.salt_vs_pepper))

        # salt
        ys = np.random.randint(0, img_np.shape[0], num_salt)
        xs = np.random.randint(0, img_np.shape[1], num_salt)
        noisy[ys, xs] = 255
        # pepper
        ys = np.random.randint(0, img_np.shape[0], num_pepper)
        xs = np.random.randint(0, img_np.shape[1], num_pepper)
        noisy[ys, xs] = 0
        return noisy

        if self.noise_type == "scatter_high":
            return self._scatter_high_values(img_np)

        return img_np
    def _add_noise(self, img_np: np.ndarray) -> np.ndarray:
        noisy = img_np.copy()
        if "gaussian" in self.noise_type:
            noisy = self._add_gaussian_noise(noisy)
        if "salt_pepper" in self.noise_type:
            noisy = self._add_salt_pepper_noise(noisy)
        if "scatter_high" in self.noise_type:
            noisy = self._scatter_high_values(noisy)
        return noisy
    # ────────────────────────────────────────────────────────────────────────
    #  Core image-generation routine
    # ────────────────────────────────────────────────────────────────────────
    # def _render_signature(self, text: str) -> np.ndarray:
    #     """Render a single word onto an RGBA canvas, return grayscale np.uint8.""" 
    #     font_path    = random.choice(self.fonts)
    #     font_size    = self.font_size
    #     image_size   = self.image_size          # e.g. (800, 100)
    #     text_color   = 0                      # white ink on black background
    #     bg_color     = 255                        # pure black background
    #     rotation_deg = 5
    #     # ---- canvas: single‑channel (“L”) image ----
    #     img = Image.new("L", image_size, bg_color)
    #     draw = ImageDraw.Draw(img)
    #     font = ImageFont.truetype(str(font_path), font_size)

    #     # heuristic starting point
    #     x_cursor = random.randint(int(font_size * 0.1), int(font_size * 5))
    #     y_base   = int(image_size[1] * 0.4)
    #     for ch in text:
    #         bbox   = font.getbbox(ch)
    #         char_w = bbox[2] - bbox[0]
    #         char_h = bbox[3] - bbox[1]
    #         mask = Image.new("L", (char_w, char_h), bg_color)
    #         ImageDraw.Draw(mask).text((-bbox[0], -bbox[1]), ch, font=font, fill=text_color,stroke_fill=text_color,embedded_color=text_color)
    #         mask = mask.rotate(random.randint(-rotation_deg, rotation_deg), resample=Image.BICUBIC, expand=True,fillcolor=bg_color        )
    #         img.paste(mask, (x_cursor, y_base - mask.size[1] // 2), mask)
    #         x_cursor +=  random.randint(int(font_size * 0.1), int(font_size * 0.4))
    #     img_np = np.array(img, dtype=np.uint8)
    #     img_np = self._add_noise(img_np)
    #     return img_np
    
    
    # 1. Simulate variable stroke width (pen pressure)
    def _apply_stroke(self, ch, font, bbox, stroke_width, text_color, bg_color):
        char_w = bbox[2] - bbox[0]
        char_h = bbox[3] - bbox[1]
        mask = Image.new("L", (char_w + 4, char_h + 4), bg_color)
        draw_mask = ImageDraw.Draw(mask)
        draw_mask.text((-bbox[0] + 2, -bbox[1] + 2), ch, font=font,
                    fill=text_color, stroke_width=stroke_width,stroke_fill=text_color)
        return mask

    # 2. Apply shear (slant transformation)
    def _apply_shear(self, img, shear_amount, bg_color,text_color):
        matrix = (1, shear_amount, 0, 0, 1, 0)
        return img.transform(img.size, Image.AFFINE, matrix, fillcolor=bg_color)

    # 3. Apply rotation (for handwritten variation)
    def _apply_rotation(self, img, rotation_deg, bg_color,text_color):
        return img.rotate(
            random.randint(-rotation_deg, rotation_deg),
            resample=Image.BICUBIC,
            expand=True,
            fillcolor=bg_color
        )

    # 4. Compute vertical jitter (baseline sine wave)
    def _baseline_y(self, i, image_height, font_size, total_chars):
        wave_amplitude = font_size * 0.05
        wave_frequency = 2 * math.pi / max(1, total_chars)
        return int(image_height * 0.4 + wave_amplitude * math.sin(i * wave_frequency))

    # 5. Cursive overlap spacing
    def _next_x_cursor(self, mask_width, font_size):
        spacing = mask_width - random.randint(int(font_size * 0.15), int(font_size * 0.4))
        return max(5, spacing)

    # Main signature rendering method
    def _render_signature(self, text: str) -> np.ndarray:
        font_path    = random.choice(self.fonts)
        font_size    = self.font_size
        image_size   = self.image_size
        text_color   = 225     # black
        bg_color     = 0   # white
        rotation_deg = 5

        font = ImageFont.truetype(str(font_path), font_size)
        img = Image.new("L", image_size, bg_color)
        draw = ImageDraw.Draw(img)

        x_cursor = random.randint(int(font_size * .5), int(font_size * 2))

        for i, ch in enumerate(text):
            bbox = font.getbbox(ch)
            # Simulate pressure
            stroke_width = random.randint(1, 3)
            mask = self._apply_stroke(ch, font, bbox, stroke_width, text_color, bg_color)
            # Slant
            shear_amt = random.uniform(-0.25, 0.25)
            mask = self._apply_shear(mask, shear_amt, bg_color, text_color)
            # Rotate
            mask = self._apply_rotation(mask, rotation_deg, bg_color, text_color)
            # Baseline jitter
            y_base = self._baseline_y(i, image_size[1], font_size, len(text))
            # Paste character
            img.paste(mask, (x_cursor, y_base - mask.size[1] // 2), mask)
            # Overlap (cursive effect)
            x_cursor += self._next_x_cursor(mask.size[0], font_size)
        img=img.convert("L")
        img=np.array(img, dtype=np.uint8)
        img=self._add_noise(img)
        return 255-np.array(img, dtype=np.uint8)


    # ────────────────────────────────────────────────────────────────────────
    #  Public helpers
    # ────────────────────────────────────────────────────────────────────────
    def remove_all(self) -> None:
        """Delete every PNG inside self.signatures_folder."""
        for f in Path(self.signatures_folder).glob("*.png"):
            f.unlink(missing_ok=True)

    def create_signature(self,text, file_nos=0, show: bool = False, ):
        file_path = f"{self.signatures_folder}{text}_{file_nos}.png"
        image_np = self._render_signature(self.name)
        Image.fromarray(image_np).save(file_path)
        if show:
            plt.figure(figsize=(4, 1))
            plt.imshow(image_np, cmap="gray")
            plt.axis("off")
            plt.tight_layout()
            plt.show()
        return str(file_path), image_np

    def create_multiple(self, show: bool = False, ):
        """Generate several variants of the base name."""
        base_variants=[]
        for p in self.pct_changes:
            num_changes=int(len(self.name)*p)
            base_variants=base_variants+self._similar_strings(self.name, num_variants=self.num_variants, num_changes=num_changes, )
        for text in base_variants:
            for k in range(np.random.random_integers(3,7)):
                self.create_signature(text, file_nos=k)            
        self.base_variants=base_variants
        return base_variants
        files = []
        for variant in self.base_variants:
            path, _ = self.create_signature(file_nos=f"{variant}.png", show=show)
            files.append(path)
        return files

    def ground_truth_score(self) -> pd.DataFrame:
        """Compute Levenshtein similarity between every generated PNG pair."""
        signatures=[filename for filename in os.listdir(self.signatures_folder)]  
        signatures=pd.DataFrame([[u,v,self._levenshtein_index(u.split("_")[0],v.split("_")[0])] for u in  signatures for v in  signatures])
        signatures.columns=["Signature_1", "Signature_2", "similarity"]
        return signatures


# ═════════════════════════════════════════════════════════════════════════════
#  2.  Siamese network for signature verification
# ═════════════════════════════════════════════════════════════════════════════
class Signature_Siamese_Network:
    """
    Flexible Siamese architecture. Call with network_name =
        'base_cnn'  → small 3-layer CNN
        'resnet'    → ResNet-50 embedding (frozen)
        'transformer'→ ViT-style  patch encoder
    """

    # ────────────────────────────────────────────────────────────────────────
    #  Base encoders
    # ────────────────────────────────────────────────────────────────────────
    def _build_base_cnn(self, input_shape) -> Model:
        inputs = Input(shape=input_shape)
        x = layers.Conv2D(64, (3, 3), activation="relu")(inputs)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(128, (3, 3), activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        return Model(inputs, x, name="base_cnn")

    def _build_resnet_embedding(self, input_shape) -> Model:
        base = ResNet50(weights=None, include_top=False, input_shape=input_shape)
        base.trainable = False
        inputs = Input(shape=input_shape)
        x = base(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        return Model(inputs, x, name="resnet")

    def _build_transformer(self, input_shape) -> Model:
        patch_size = 20
        projection_dim = 64
        num_heads = 4
        ff_dim = 128
        num_layers = 4

        inputs = Input(shape=input_shape)
        num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)

        # patch extraction
        patches = tf.image.extract_patches(images=tf.expand_dims(inputs, axis=0), sizes=[1, patch_size, patch_size, 1], strides=[1, patch_size, patch_size, 1], rates=[1, 1, 1, 1], padding="VALID", )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, (-1, num_patches, patch_dims))
        x = layers.Dense(projection_dim)(patches)

        # transformer encoder blocks
        for _ in range(num_layers):
            x1 = layers.LayerNormalization()(x)
            attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim
            )(x1, x1)
            x2 = layers.Add()([x, attn_out])
            x3 = layers.LayerNormalization()(x2)
            x3 = layers.Dense(ff_dim, activation="relu")(x3)
            x3 = layers.Dense(projection_dim)(x3)
            x = layers.Add()([x2, x3])

        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        return Model(inputs, x, name="transformer")

    # ────────────────────────────────────────────────────────────────────────
    #  Siamese wrapper
    # ────────────────────────────────────────────────────────────────────────
    def _build_siamese(self) -> Model:
        lookup = {
            "base_cnn": self._build_base_cnn, "resnet": self._build_resnet_embedding, "transformer": self._build_transformer, }
        if self.network_name not in lookup:
            raise ValueError(f"network_name must be one of {list(lookup)}")
        encoder = lookup[self.network_name](self.input_shape)

        input_a = Input(shape=self.input_shape)
        input_b = Input(shape=self.input_shape)
        feat_a = encoder(input_a)
        feat_b = encoder(input_b)

        diff = layers.Lambda(lambda t: tf.abs(t[0] - t[1]))([feat_a, feat_b])
        output = layers.Dense(1, activation="sigmoid")(diff)

        return Model(inputs=[input_a, input_b], outputs=output)

    # ────────────────────────────────────────────────────────────────────────
    #  Public API
    # ────────────────────────────────────────────────────────────────────────
    def __init__(self, input_shape: Tuple[int, int, int] = (32, 400, 1),                 
                 target_folder: str = target_folder, 
                 signatures_folder: str = "Signatures/", 
                 loss: str = "binary_crossentropy", network_name: str = "base_cnn",  ):
        self.target_folder = target_folder
        self.signatures_folder = target_folder+signatures_folder
        self.input_shape = input_shape
        self.loss = loss
        self.network_name = network_name
        # self.signatures_folder = Path(signatures_folder)
        self.model = self._build_siamese()
        self.model.compile(loss=loss, optimizer=Adam(), metrics=["accuracy"])
        # self.model.summary()
        
        self.Signature_1 = "Signature_1"
        self.Signature_2 = "Signature_2"
        self.y = "y"
        self.loss = loss
        self.loss = loss
        self.loss = loss
    def save(self):
        config = {
        "input_shape": self.input_shape,
        "network_name": self.network_name,
        "target_folder": self.target_folder,
        "signatures_folder": self.signatures_folder,
        "loss": self.loss,
        "Signature_1": self.Signature_1,
        "Signature_2": self.Signature_2,
        "y": self.y, 
        }
        joblib.dump(config,self.target_folder+"siamese.data")
        self.model.save(self.target_folder+"siamese.h5")

    def load(self):

      
        config=joblib.load(self.target_folder+"siamese.data")
        self.input_shape=config['input_shape']
        self.network_name=config['network_name']
        self.target_folder=config['target_folder']
        self.signatures_folder=config['signatures_folder']
        self.loss=config['loss']
        self.Signature_1=config['Signature_1']
        self.Signature_2=config['Signature_2']
        self.y=config['y']        
        self.model = self.model.load(self.target_folder+"siamese.h5")



    # ────────────────────────────────────────────────────────────────────────
    #  Training helpers
    # ────────────────────────────────────────────────────────────────────────
    def _train_val_split(self, images: List[np.ndarray], labels: List[int], frac=0.2):
        idx = np.arange(len(images))
        np.random.shuffle(idx)
        split = int(len(idx) * (1 - frac))
        train_idx, val_idx = idx[:split], idx[split:]
        X_train, y_train = np.array(images)[train_idx], np.array(labels)[train_idx]
        X_val, y_val = np.array(images)[val_idx], np.array(labels)[val_idx]
        return X_train, y_train, X_val, y_val

    def load_gray(self,path):
        return np.array(Image.open(path).convert("L"), dtype=np.uint8)
    def return_images(self,Train):
        X=None
        y=None
        if Train is not None:
            img_A = np.stack([self.load_gray(self.signatures_folder+f) for f in Train[self.Signature_1]],axis=0) / 255.0
            img_B = np.stack([self.load_gray(self.signatures_folder+f) for f in Train[self.Signature_2]],axis=0) / 255.0
            y = Train[self.y].values.reshape(-1, 1)
            X=[img_A,img_B]
        return X,y
    
    def fit(self, Train: pd.DataFrame,Valid: pd.DataFrame=None, epochs: int = 10, batch_size: int = 32, **kw):
        """
        `pairs` must be a DataFrame with columns ["file_A", "file_B", "label"]
        where label = 1 for "same" and 0 for "different".
        """
        X,y=self.return_images(Train)
        X_v,y_v=self.return_images(Valid)
        
        if Valid is not None:
            X_v,y_v=self.return_images(Valid)
            history = self.model.fit(X,y, validation_data=(X_v,y_v), epochs=epochs, batch_size=batch_size, **kw, )
        else:
            history = self.model.fit(X,y,   epochs=epochs, batch_size=batch_size, **kw, )


        return history
    def predict(self, data: pd.DataFrame,):
        X,y = self.return_images(data)
        pred = self.model.predict(X,)
        return pred.ravel()
    # ────────────────────────────────────────────────────────────────────────
    #  Evaluation plots
    # ────────────────────────────────────────────────────────────────────────
    @staticmethod
    def plot_history(history: tf.keras.callbacks.History) -> None:
        acc = history.history.get("accuracy", [])
        val_acc = history.history.get("val_accuracy", [])
        loss = history.history.get("loss", [])
        val_loss = history.history.get("val_loss", [])
        epochs = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))

        # accuracy
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, "bo-", label="Training Acc")
        if val_acc:
            plt.plot(epochs, val_acc, "ro-", label="Validation Acc")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Acc")
        plt.grid(True)
        plt.legend()

        # loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, "bo-", label="Training Loss")
        if val_loss:
            plt.plot(epochs, val_loss, "ro-", label="Validation Loss")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    # ────────────────────────────────────────────────────────────────────────
    #  Fourier helpers (handy for visual debugging)
    # ────────────────────────────────────────────────────────────────────────
    @staticmethod
    def fourier_transform(image_np: np.ndarray) -> np.ndarray:
        f = np.fft.fft2(image_np)
        return np.fft.fftshift(f)

    @staticmethod
    def inverse_fourier_transform(fshift: np.ndarray) -> np.ndarray:
        ishift = np.fft.ifftshift(fshift)
        img = np.fft.ifft2(ishift)
        return np.abs(img).astype(np.uint8)



