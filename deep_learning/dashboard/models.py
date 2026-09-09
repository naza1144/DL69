from django.db import models


class MLPModel(models.Model):
    """
    Recording for a trained MLP (wk10: Linear(2,8)->ReLU->Linear(8,1)->Sigmoid).

    Captures the dataset geometry that generated the training points plus
    the persisted model artifact location.

    Frontend source (dashboard/templates/dashboard/training.html):
      - Label rule: op ∈ {">", "<"}  where label = (x*y > 0) vs (x*y < 0)
      - Center X / Center Y: translation applied after rotation
      - Rotation: degrees counter-clockwise applied before translation
    """

    class LabelRule(models.TextChoices):
        GT = ">", "x·y > 0 (green if >)"
        LT = "<", "x·y < 0 (green if <)"

    label_rule = models.CharField(
        max_length=2,
        choices=LabelRule.choices,
        default=LabelRule.GT,
        help_text='Label rule for product dataset: ">" means label=1 if x*y>0 else 0; "<" inverts.',
    )
    center_x = models.FloatField(
        default=0.0,
        help_text="Dataset center X (translation after rotation).",
    )
    center_y = models.FloatField(
        default=0.0,
        help_text="Dataset center Y (translation after rotation).",
    )
    rotation = models.FloatField(
        default=0.0,
        help_text="Rotation in degrees (counter-clockwise) applied to (rx,ry) before shifting by center.",
    )

    # artifact location — e.g. "models/mlp_{id}.pt" or absolute path
    file_path = models.CharField(
        max_length=512,
        blank=True,
        default="",
        help_text="Filesystem path to the persisted torch model (e.g. torch.save state_dict).",
    )

    create_on = models.DateTimeField(
        auto_now_add=True,
        help_text="Creation timestamp.",
    )
    update_on = models.DateTimeField(
        auto_now=True,
        help_text="Last update timestamp.",
    )

    class Meta:
        ordering = ["-create_on"]
        verbose_name = "MLP model"
        verbose_name_plural = "MLP models"

    def __str__(self) -> str:
        return (
            f"MLPModel id={self.pk} "
            f"rule={self.label_rule} "
            f"center=({self.center_x},{self.center_y}) "
            f"rot={self.rotation}°"
        )

    @property
    def label_rule_display(self) -> str:
        return f"x·y {self.label_rule} 0"
