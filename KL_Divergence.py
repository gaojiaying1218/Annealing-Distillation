import torch
import torch.nn.functional as F


def kd_loss(student_logits, teacher_logits, temperature=2.0):
    """
    Compute the knowledge distillation loss.

    Args:
        student_logits (Tensor): Logits from the student model. Shape (batch_size, num_classes).
        teacher_logits (Tensor): Logits from the teacher model. Shape (batch_size, num_classes).
        temperature (float): Temperature for scaling the logits.

    Returns:
        Tensor: Knowledge distillation loss.
    """
    # Compute softmax with temperature scaling for both student and teacher
    student_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)

    # Compute the KL divergence loss
    kd_loss_value = F.kl_div(student_probs, teacher_probs, reduction='batchmean')

    # Scale the loss by temperature squared
    kd_loss_value *= (temperature ** 2)

    return kd_loss_value


# Example usage
student_preds = torch.randn(10, 5)  # Example student logits (batch_size, num_classes)
teacher_preds = torch.randn(10, 5)  # Example teacher logits (batch_size, num_classes)
temperature = 2.0

# Compute the distillation loss
distillation_loss = kd_loss(student_preds, teacher_preds, temperature)
print(distillation_loss.item())
