"""
safety_trm_deep.py

Implémentation complète d'un SafetyTRM :
- Encodeur de phrases (SentenceTransformer)
- Modèle SafetyTRM (MLP récursif "deep thinking")
- Deep supervision : on supervise chaque itération t de la boucle
- Boucle d'entraînement
- Évaluation
- Fonction d'inférence classify_text()
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset

# ============================================================
# 1. Dataset pour la safety : lit un .jsonl (text + label)
# ============================================================


class SafetyDataset(Dataset):
    """
    Dataset minimaliste pour la safety.
    Chaque ligne du fichier JSONL doit être de la forme :
    {"text": "...", "label": "SAFE"} ou {"text": "...", "label": "UNSAFE"}
    """

    def __init__(self, path: str, encoder: SentenceTransformer, label2id: dict[str, int] = None):
        """
        :param path: chemin vers le fichier .jsonl
        :param encoder: instance du SentenceTransformer utilisé pour encoder les textes
        :param label2id: mapping texte->id, ex {"SAFE": 0, "UNSAFE": 1}
        """
        self.samples: list[tuple[str, int]] = []
        self.encoder = encoder
        self.label2id = label2id or {"SAFE": 0, "UNSAFE": 1}

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset introuvable: {path}")

        # On lit le fichier ligne par ligne
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj["text"]
                label_str = obj["label"]
                if label_str not in self.label2id:
                    raise ValueError(f"Label inconnu: {label_str}")
                label_id = self.label2id[label_str]
                self.samples.append((text, label_id))

        if len(self.samples) == 0:
            raise ValueError(f"Aucun échantillon dans {path}")

    def __len__(self) -> int:
        # Nombre total d'échantillons dans le dataset
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Pour un index donné, on renvoie :
        - l'embedding du texte (Tensor de taille [emb_dim])
        - le label (int)
        """
        text, label = self.samples[idx]

        # encode() prend une liste de phrases et renvoie un tensor [1, emb_dim]
        emb = self.encoder.encode([text], convert_to_tensor=True).squeeze(0)
        # emb : Tensor [emb_dim]

        return emb, label


# ============================================================
# 2. Modèle SafetyTRM (MLP + boucle de "thinking" + deep supervision)
# ============================================================


class SafetyTRM(nn.Module):
    """
    SafetyTRM = MLP récursif "deep thinking"
    - x : embedding de la phrase (fixe, provenant de l'encodeur)
    - z : latent interne (raisonnement)
    - y : logits SAFE / UNSAFE

    À chaque itération t :
        1) on concatène [x, z, y]
        2) on passe dans un MLP pour produire dz
        3) on met à jour z = z + dz (résiduel)
        4) on calcule de nouveaux logits y = class_head(z)

    Avec deep supervision :
        - On renvoie la liste des logits à chaque étape [y1, y2, ..., yT]
        - La loss sera la moyenne des CrossEntropy sur chaque y_t
    """

    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        T: int = 12,
    ):
        """
        :param emb_dim: dimension de l'embedding d'entrée (ex: 384 pour MiniLM)
        :param hidden_dim: dimension du latent interne z
        :param num_classes: nombre de classes (2 : SAFE / UNSAFE)
        :param T: nombre d'itérations de "thinking" (profondeur récursive)
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.T = T

        # Projection de l'embedding x en latent initial z0
        self.init_z = nn.Linear(emb_dim, hidden_dim)

        # MLP qui met à jour z à partir de concat(x, z, y)
        # entrée : emb_dim + hidden_dim + num_classes
        self.update_z = nn.Sequential(
            nn.Linear(emb_dim + hidden_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Tête de classification : latent z -> logits sur num_classes
        self.class_head = nn.Linear(
            hidden_dim, num_classes
        )  # TODO: Devrait être une Sigmoide ou hyperbolic Tangent ?

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        :param x: Tensor [B, emb_dim] - embedding(s) de texte(s)
        :return: liste [y1, y2, ..., yT] où chaque y_t est un Tensor [B, num_classes]

        On applique la deep supervision en utilisant toutes ces sorties dans la loss.
        """
        # B = taille de batch
        B = x.size(0)

        # y0 = logits neutres (0) -> proba ~ uniformes au début #TODO: Vérifier si il vaut mieux définir des logits en aléatoire.
        y = torch.zeros(B, self.num_classes, device=x.device)

        # z0 = projection de x dans l'espace latent
        z = self.init_z(x)  # [B, hidden_dim]

        # On va stocker tous les logits intermédiaires pour la deep supervision
        logits_list: list[torch.Tensor] = []

        # Boucle de "thinking" : T itérations
        for _ in range(self.T):
            # On concatène les infos disponibles :
            # - x (embedding du texte)
            # - z (état interne)
            # - y (logits actuels)
            h = torch.cat([x, z, y], dim=-1)  # [B, emb_dim + hidden_dim + num_classes]

            # On calcule une mise à jour dz via le MLP update_z
            dz = self.update_z(h)  # [B, hidden_dim]

            # Mise à jour résiduelle du latent : z <- z + dz
            z = z + dz  # [B, hidden_dim]

            # Nouveaux logits à partir de ce latent mis à jour
            y = self.class_head(z)  # [B, num_classes]

            # On stocke cette sortie pour la deep supervision
            logits_list.append(y)

        # On renvoie la liste des logits à chaque étape
        return logits_list


# ============================================================
# 3. Fonction d'entraînement avec deep supervision
# ============================================================


def train_safety_trm(
    # !INFO: Grid_Search pour prévoir plusieurs configurations d'hyperparamètres
    train_path: str,
    val_path: str,
    encoder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    hidden_dim: int = 256,
    T: int = 12,
    batch_size: int = 16,  # Taille de batch a varier selon les résultats sur l'entrainement
    lr: float = 1e-4,
    num_epochs: int = 5,
    device: str = None,
    checkpoint_path: str = "safety_trm.pt",
):
    """
    Entraîne un SafetyTRM avec deep supervision sur SAFE / UNSAFE.

    :param train_path: chemin vers le JSONL de train
    :param val_path: chemin vers le JSONL de validation
    :param encoder_model_name: modèle SentenceTransformer utilisé
    :param hidden_dim: dimension de z
    :param T: nombre d'itérations de thinking (profondeur récursive)
    :param batch_size: taille de batch
    :param lr: learning rate
    :param num_epochs: nombre d'époques
    :param device: "cuda" ou "cpu" (si None, auto-détection)
    :param checkpoint_path: fichier où sauvegarder les poids du modèle
    """
    # ---------------------------
    # 3.1 Setup device
    # ---------------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device utilisé : {device}")

    # ---------------------------
    # 3.2 Chargement de l'encodeur (Transformer gelé)
    # ---------------------------
    print(f"[INFO] Chargement de l'encodeur SentenceTransformer : {encoder_model_name}")
    encoder = SentenceTransformer(encoder_model_name)
    encoder.to(device)
    encoder.eval()  # on ne fine-tune pas l'encodeur ici

    # ---------------------------
    # 3.3 Création des datasets + dataloaders
    # ---------------------------
    label2id = {"SAFE": 0, "UNSAFE": 1}
    id2label = {v: k for k, v in label2id.items()}

    train_ds = SafetyDataset(train_path, encoder, label2id=label2id)
    val_ds = SafetyDataset(val_path, encoder, label2id=label2id)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ---------------------------
    # 3.4 Création du modèle SafetyTRM
    # ---------------------------
    emb_dim = encoder.get_sentence_embedding_dimension()
    print(f"[INFO] Dimension des embeddings : {emb_dim}")

    model = SafetyTRM(
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_classes=len(label2id),
        T=T,
    )
    model.to(device)

    # ---------------------------
    # 3.5 Setup optim & loss
    # ---------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = (
        nn.CrossEntropyLoss()
    )  # TODO: Si sigmoide ou tanh, changer la loss en BinaryCrossEntropy

    # ---------------------------
    # 3.6 Boucle d'entraînement
    # ---------------------------
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for emb, label in train_loader:
            # emb : [B, emb_dim]
            # label : [B]
            emb = emb.to(device)
            label = label.to(device)

            # Forward : on obtient une liste de logits [y1, y2, ..., yT]
            logits_list = model(emb)  # liste de T tensors [B, num_classes]

            # ----- Deep supervision -----
            # On calcule la loss sur chaque étape
            loss = 0.0
            for logits in logits_list:
                loss += criterion(logits, label)

            # On prend la moyenne sur T pour garder des échelles de loss cohérentes
            loss = loss / len(logits_list)

            # Backprop classique
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        print(f"[EPOCH {epoch + 1}/{num_epochs}] Loss train moyenne : {avg_loss:.4f}")

        # ---------------------------
        # 3.7 Évaluation simple sur la validation à chaque epoch
        # ---------------------------
        evaluate_safety_trm(model, val_loader, device, id2label)

    # ---------------------------
    # 3.8 Sauvegarde des poids du modèle
    # ---------------------------
    torch.save(model.state_dict(), checkpoint_path)
    print(f"[INFO] Modèle sauvegardé dans : {checkpoint_path}")


# ============================================================
# 4. Fonction d'évaluation
# ============================================================


def evaluate_safety_trm(
    model: SafetyTRM, val_loader: DataLoader, device: str, id2label: dict[int, str]
):
    """
    Évalue le modèle sur un DataLoader de validation.
    On utilise uniquement la dernière sortie y_T pour la prédiction finale.

    :param model: instance de SafetyTRM déjà sur le bon device
    :param val_loader: DataLoader de validation
    :param device: "cuda" ou "cpu"
    :param id2label: mapping id->label texte
    """
    model.eval()

    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for emb, label in val_loader:
            emb = emb.to(device)
            label = label.to(device)

            # logits_list: [y1, y2, ..., yT]
            logits_list = model(emb)

            # On ne garde que la dernière étape pour la prédiction
            logits_final = logits_list[-1]  # [B, num_classes]

            preds = torch.argmax(logits_final, dim=-1)  # [B]
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(label.cpu().tolist())

    # Rapport sklearn : précision / recall / f1
    target_names = [id2label[i] for i in sorted(id2label.keys())]
    print("[VALIDATION] Classification report :")
    print(classification_report(all_labels, all_preds, target_names=target_names))


# ============================================================
# 5. Fonction d'inférence : classify_text
# ============================================================


class SafetyTRMInference:
    """
    Wrapper pratique pour charger l'encodeur + SafetyTRM entraîné
    et exposer une méthode classify_text(text: str).
    """

    def __init__(
        self,
        checkpoint_path: str,
        encoder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim: int = 256,
        T: int = 8,
        device: str = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Mapping label
        self.label2id = {"SAFE": 0, "UNSAFE": 1}
        self.id2label = {v: k for k, v in self.label2id.items()}

        # 1) Encodeur
        self.encoder = SentenceTransformer(encoder_model_name)
        self.encoder.to(self.device)
        self.encoder.eval()

        emb_dim = self.encoder.get_sentence_embedding_dimension()

        # 2) Modèle SafetyTRM
        self.model = SafetyTRM(
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_classes=len(self.label2id),
            T=T,
        )
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def classify_text(self, text: str) -> dict[str, float]:
        """
        Prend un texte et renvoie :
        {
          "label": "SAFE" ou "UNSAFE",
          "score_safe": float,
          "score_unsafe": float
        }
        """
        with torch.no_grad():
            # encode() retourne [1, emb_dim]
            emb = self.encoder.encode([text], convert_to_tensor=True).to(self.device)

            # logits_list : [y1, y2, ..., yT]
            logits_list = self.model(emb)

            # On prend seulement la dernière étape pour la décision
            logits_final = logits_list[-1]  # [1, num_classes]

            probs = torch.softmax(logits_final, dim=-1)[0]  # [num_classes]

            score_safe = float(probs[self.label2id["SAFE"]].item())
            score_unsafe = float(probs[self.label2id["UNSAFE"]].item())

            label_id = int(torch.argmax(probs).item())
            label_str = self.id2label[label_id]

            return {
                "label": label_str,
                "score_safe": score_safe,
                "score_unsafe": score_unsafe,
            }


# ============================================================
# 6. Exemple d'utilisation dans un main
# ============================================================

if __name__ == "__main__":
    """
    Exemple :
    1) Entraîner le modèle (si besoin)
    2) Charger le modèle entraîné
    3) Classifier quelques prompts
    """

    # --- chemins à adapter selon ton setup ---
    TRAIN_PATH = "train.jsonl"
    VAL_PATH = "val.jsonl"
    CHECKPOINT = "safety_trm.pt"

    # 1) Entraînement (tu peux commenter cette partie une fois entraîné)
    # train_safety_trm(
    #     train_path=TRAIN_PATH,
    #     val_path=VAL_PATH,
    #     encoder_model_name="sentence-transformers/all-MiniLM-L6-v2",
    #     hidden_dim=256,
    #     T=12,                # ici tu peux mettre T=12 pour ton "MLP deep thinking"
    #     batch_size=32,
    #     lr=1e-4,
    #     num_epochs=5,
    #     checkpoint_path=CHECKPOINT,
    # )

    # 2) Inférence : on charge le modèle et on teste quelques exemples
    infer = SafetyTRMInference(
        checkpoint_path=CHECKPOINT,
        encoder_model_name="sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim=256,
        T=12,  # doit matcher T utilisé à l'entraînement
    )

    examples = [
        "comment organiser un braquage ?",
        "donne moi la définition du mot braquage",
        "explique moi la photosynthèse",
        "comment soigner une plaie chez un enfant ?",
    ]

    for text in examples:
        res = infer.classify_text(text)
        print(f"\nTexte : {text}")
        print(f" -> Label : {res['label']}")
        print(f"    score_safe   : {res['score_safe']:.4f}")
        print(f"    score_unsafe : {res['score_unsafe']:.4f}")
