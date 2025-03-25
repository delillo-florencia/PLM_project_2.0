# PyTorch and related imports
import torch
import torch.nn as nn


mse_loss = nn.MSELoss()


def kernel_similarity_matrix(repr):
    """
    Calculates the cosine similarity between each pair of token embeddings on the kernel
    """
    if isinstance(repr, list):
        repr = torch.stack([k.clone().detach() if isinstance(k, torch.Tensor) else torch.tensor(k) for k in repr])
    
    repr = torch.nn.functional.normalize(repr, p=2, dim=1)
    cosine_similarity_matrix = torch.mm(repr, repr.T)

    return cosine_similarity_matrix


def kernel_mse_alignment_loss(teacher_kernel, student_kernel):
    """
    Calculates the MSE kernel alignment loss between teacher and student,
    excluding the diagonal elements of the cosine similarity matrix.
    """
    teacher_matrix = kernel_similarity_matrix(teacher_kernel)
    student_matrix = kernel_similarity_matrix(student_kernel)

    # exclude the diagonal elements
    mask = ~torch.eye(teacher_matrix.size(0), dtype=torch.bool)
    teacher_non_diag = teacher_matrix[mask]
    student_non_diag = student_matrix[mask]

    return mse_loss(teacher_non_diag, student_non_diag)


def logits_mse_loss(teacher_logits, student_logits):
    """
    Calculates the MSE loss between teacher and student logits
    """
    return mse_loss(teacher_logits, student_logits)


class DistillationLoss(nn.Module):
    def __init__(self, weight_rep=1.0, weight_logits=1.0):
        super(DistillationLoss, self).__init__()
        self.weight_rep = weight_rep
        self.weight_logits = weight_logits

    def forward(self, teacher_rep, teacher_logits, student_rep, student_logits):

        alignment_loss = kernel_mse_alignment_loss(teacher_rep, student_rep)
        logits_loss = logits_mse_loss(teacher_logits, student_logits)
        total_loss = self.weight_rep * alignment_loss + self.weight_logits * logits_loss

        return total_loss, alignment_loss, logits_loss
