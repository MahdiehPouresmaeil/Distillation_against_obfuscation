import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
from itertools import chain
import gc

# from tests_notebooks.ResToTransformer import target_layer


# --- 1. Module Adapter (Pour aligner Student -> Teacher) ---
class Adapter(nn.Module):
    """
    Adapte la sortie du Student pour correspondre à la taille (Channels) du Teacher.
    Approche 'FitNets'.
    """

    def __init__(self, s_shape, t_shape):
        super(Adapter, self).__init__()

        # s_shape et t_shape : [Batch, Channel, H, W] (Conv) ou [Batch, Features] (Linear)
        self.mode = 'conv' if len(t_shape) == 4 else 'linear'

        if self.mode == 'conv':
            s_channels = s_shape[1]
            t_channels = t_shape[1]

            # Projection 1x1 pour aligner les canaux
            # (On pourrait utiliser 3x3 padding 1 pour plus de capacité)
            self.layer_st = nn.Sequential(
                nn.Conv2d(s_channels, t_channels, kernel_size=1, stride=1, padding=0, bias=True),
                # nn.BatchNorm2d(t_channels),
                # nn.ReLU()
            )
            self.layer_ts = nn.Sequential(
                nn.Conv2d( t_channels,s_channels, kernel_size=1, stride=1, padding=0, bias=True),
                # nn.BatchNorm2d(t_channels),
                # nn.ReLU()
            )
        else:
            s_features = s_shape[1]
            t_features = t_shape[1]
            self.layer = nn.Sequential(
                nn.Linear(s_features, t_features, bias=False),
                # nn.ReLU()
            )

    def forward(self, x_s, x_t):
        return self.layer_st(x_s), self.layer_ts(x_t)


class ResponseBasedKDLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.9):
        super(ResponseBasedKDLoss, self).__init__()
        self.T = temperature
        self.alpha = alpha
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        soft_student = F.log_softmax(student_logits / self.T, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.T, dim=1)
        distillation_loss = self.kl_div_loss(soft_student, soft_teacher) * (self.T ** 2)
        student_loss = self.ce_loss(student_logits, labels)
        total_loss = (self.alpha * distillation_loss) + ((1 - self.alpha) * student_loss)
        return total_loss


class WeightMatcher(nn.Module):
    """Handles any shape mismatch"""

    def __init__(self, s_shape, t_shape, method='interpolate'):
        super().__init__()
        self.s_shape = s_shape
        self.t_shape = t_shape
        self.method = method

        if method == 'project':
            s_out, s_in, s_kh, s_kw = s_shape
            t_out, t_in, t_kh, t_kw = t_shape
            self.proj_out = nn.Linear(s_out, t_out)
            self.proj_in = nn.Linear(s_in, t_in)
            self.proj_k = nn.Linear(s_kh * s_kw, t_kh * t_kw)

    def forward(self, s_weight, t_weight):
        if self.method == 'statistics':
            return self._statistics_loss(s_weight, t_weight)
        elif self.method == 'interpolate':
            s_matched = self._interpolate(s_weight)
            return F.mse_loss(s_matched, t_weight)
        elif self.method == 'project':
            s_projected = self._project(s_weight)
            return F.mse_loss(s_projected, t_weight)

    def _statistics_loss(self, s_weight, t_weight):
        return (F.mse_loss(s_weight.mean(), t_weight.mean()) +
                F.mse_loss(s_weight.std(), t_weight.std()))

    def _interpolate(self, s_weight):
        """
        s_weight: [s_out, s_in, s_kh, s_kw]
        output:   [t_out, t_in, t_kh, t_kw]
        """
        t_out, t_in, t_kh, t_kw = self.t_shape

        # Step 1: Resize kernel (s_kh, s_kw) → (t_kh, t_kw)
        # Input: [s_out, s_in, s_kh, s_kw] - already correct format [N, C, H, W]
        x = F.interpolate(s_weight, size=(t_kh, t_kw), mode='bilinear', align_corners=False)
        # Output: [s_out, s_in, t_kh, t_kw]

        # Step 2: Resize channels (s_out, s_in) → (t_out, t_in)
        # Need to reshape to [N, C, H, W] where H, W are the channel dimensions
        x = x.permute(2, 3, 0, 1)  # [t_kh, t_kw, s_out, s_in]

        # Reshape to [t_kh * t_kw, 1, s_out, s_in] for interpolate
        batch = x.shape[0] * x.shape[1]  # t_kh * t_kw
        x = x.reshape(batch, 1, x.shape[2], x.shape[3])  # [t_kh*t_kw, 1, s_out, s_in]

        x = F.interpolate(x, size=(t_out, t_in), mode='bilinear', align_corners=False)
        # Output: [t_kh*t_kw, 1, t_out, t_in]

        x = x.reshape(t_kh, t_kw, t_out, t_in)  # [t_kh, t_kw, t_out, t_in]
        x = x.permute(2, 3, 0, 1)  # [t_out, t_in, t_kh, t_kw]

        return x

    def _project(self, s_weight):
        s_out, s_in, s_kh, s_kw = s_weight.shape
        t_out, t_in, t_kh, t_kw = self.t_shape

        x = s_weight.reshape(s_out, s_in, -1)
        x = self.proj_k(x)
        x = x.permute(0, 2, 1)
        x = self.proj_in(x)
        x = x.permute(2, 1, 0)
        x = self.proj_out(x)
        x = x.permute(2, 0, 1).reshape(t_out, t_in, t_kh, t_kw)

        return x
# --- 2. Fonction d'Entraînement Principale ---

def train_student(student, teacher, loader, temperature=4.0, lr=1e-4, epochs=10, supp=None, device="cuda",
                  extract=None, layer_name="conv4", method_name="DICTION"):


    # Gestion des noms de couches (si liste, on prend le premier/dernier pertinent)
    # target_layer = "base.layer4.1.conv2" # RESNET18   layer_name[0] if isinstance(layer_name, list) else layer_name
    # target_layer = "base.features.28"  #VGG16
    target_layer='conv4'
    # Configuration des extracteurs
    return_nodes = {target_layer: 'features'}
    extractor_student = create_feature_extractor(student, return_nodes).to(device)
    extractor_teacher = create_feature_extractor(teacher, return_nodes).to(device)

    # On gèle le Teacher
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # Analyse des dimensions pour construire l'Adapter
    print(f"Analyse des dimensions sur la couche : {target_layer}...")
    dummy_images, _ = next(iter(loader))
    dummy_images = dummy_images.to(device)

    with torch.no_grad():
        s_out = extractor_student(dummy_images)['features']
        t_out = extractor_teacher(dummy_images)['features']

    print(f"Shape Student: {s_out.shape} | Shape Teacher: {t_out.shape}")

    # Création de l'Adapter : Student -> Teacher
    adapter = Adapter(s_shape=s_out.shape, t_shape=t_out.shape).to(device)

    # Optimiseur (Student + Adapter)
    optimizer = optim.AdamW(chain(student.parameters(), adapter.parameters()), lr=lr, weight_decay=1e-4)


    # Losses
    kd_criterion = ResponseBasedKDLoss(temperature=temperature, alpha=1)
    feature_criterion = nn.MSELoss()

    lambda_feat = 1.0  # Poids de la perte de features

    student.train()
    adapter.train()
    # weight_adapter.train()

    print("--- Démarrage de la Distillation (Logits + Features + Watermark Check) ---")

    for epoch in range(epochs):

        loop = tqdm(loader, leave=True)
        # loop = tqdm(supp["key_loader"], leave=True)
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(loop):

            # gc.collect()
            # torch.cuda.empty_cache()

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # 1. Teacher (Forward sans gradient)
            with torch.no_grad():
                t_logits = teacher(images)
                t_features = extractor_teacher(images)['features']


            # 2. Student (Forward avec gradient)
            s_logits = student(images)
            s_features_raw = extractor_student(images)['features']



            # 3. Adaptation (Alignement Spatial + Canaux)
            # Interpolation si les tailles HxW diffèrent
            if s_features_raw.shape[2:] != t_features.shape[2:]:
                s_features_raw = F.interpolate(
                    s_features_raw,
                    size=t_features.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )




            # 4. Calcul des Pertes
            loss_kd = kd_criterion(s_logits, t_logits, labels)
            #if mean and var teacher and student get close, then the feature distributions are similar
            loss_feat = feature_criterion(s_features_raw.mean(dim=[0, 2, 3]), t_features.mean(dim=[0, 2, 3])) + feature_criterion(s_features_raw.var(dim=[0, 2, 3]), t_features.var(dim=[0, 2, 3]))

            loss = loss_kd #+ (1 * loss_feat)#+ (1 *loss3)
            # loss=loss3

            loss.backward()
            optimizer.step()

            # Monitoring
            train_loss += loss.item()
            _, predicted = s_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 5. Calcul de l'erreur de Tatouage (BCE)
            error_wat_val = 0.0
            ber_student = 0.0

            if extract and supp:
                # On utilise no_grad pour l'extraction afin de ne pas alourdir la mémoire
                # Attention : 'extract' utilise souvent deepcopy, ce qui peut ralentir la boucle.
                with torch.no_grad():
                    try:
                        # wat_ext est le vecteur de probabilités [0, 1] sorti par ProjMod
                        wat_ext, ber_student = extract(student, supp)

                        target_wat = supp['watermark'].to(device)
                        pred_wat = wat_ext.to(device)
                        # pred_wat=(pred_wat > 0.5).float()


                        # print(target_wat)
                        # print(pred_wat)

                        # Calcul de la BCE (error_wat)
                        error_wat_val = F.binary_cross_entropy(pred_wat, target_wat).item()
                    except Exception as e:
                        # Gestion d'erreur si dimensions incompatibles au début
                        # print(e)
                        error_wat_val = -1.0
                        ber_student = -1.0

            # Affichage
            loop.set_description(f"Epoch [{epoch}/{epochs}]")
            loop.set_postfix(
                loss=f"{train_loss / (batch_idx + 1):.4f}",
                acc=f"{100. * correct / total:.1f}%",
                ber=f"{ber_student:.4f}",
                err_w=f"{error_wat_val:.4f}" ,
                loss_kd=f"{loss_kd:.4f}",
                loss_feat=f"{loss_feat:.4f}",
                # loss3=f"{loss3:.4f}",
                # correlation=f"{correlation:.4f}",

                # Affichage de l'erreur BCE demandée
            )
        if ber_student == 0.0:
            break


    return student.state_dict()


def train_student_hufu(student, teacher, loader, temperature=4.0, lr=1e-3, epochs=3, supp=None, device="cuda", extract=None, layer_name=None, hufu_model=None,
                       selected_indexes=None, train_loader_hufu=None, config=None):
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(student.parameters(), lr=lr, weight_decay=1e-4)
    kd_criterion = ResponseBasedKDLoss(temperature=temperature, alpha=1)
    student.train()

    print("--- Démarrage de la Distillation (Logits + Features + Watermark Check) ---")
    for epoch in range(epochs):
        loop = tqdm(loader, leave=True)
        train_loss =0
        total =0
        correct =0.0
        for batch_idx, (images, labels) in enumerate(loop):
            images, labels = images.to(device), labels.to(device)
            # teacher_activations.clear()
            # student_activations.clear()

            optimizer.zero_grad()

            # Get teacher and student logits
            with torch.no_grad():
                teacher_logits= teacher(images)
                    # Get activations from fc2
                # t = teacher_activations[layer_name]

            student_logits = student(images)
            # s = student_activations[layer_name]
            loss_kd = kd_criterion(student_logits, teacher_logits, labels)
            loss = loss_kd


            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()


            error_wat_val = 0.0
            ber_student = 0.0



            loop.set_description(f"Epoch [{epoch}/{epochs}]")
            loop.set_postfix(loss=train_loss / (batch_idx + 1), acc=100. * correct / total,
                             correct_total=f"[{correct}"f"/{total}]",
                             )
        _, mse_after, _ = extract((student), (hufu_model), selected_indexes,
                                                    train_loader_hufu,
                                                    config)
        print(f"student model: Epoch [{epoch}/{epochs}]     mse_hufu_after_student: {mse_after}")


    return student