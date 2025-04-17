# PyTorch and related imports
import torch
import torch.nn as nn

mse_loss = nn.MSELoss()



def kernel_similarity_matrix(repr):
    """
    Calculates the cosine similarity between each pair of token embeddings on the kernel
    """
    if isinstance(repr, list):
        repr = torch.stack([torch.tensor(k) for k in repr])  # Convert list to tensor
    
    repr = nn.functional.normalize(repr, p=2, dim=1)
    cosine_similarity_matrix = torch.mm(repr, repr.T)

    return cosine_similarity_matrix

def kernel_mse_alignment_loss(teacher_kernel, student_kernel):
    """
    Calculates the MSE kernel alignment loss between teacher and student
    """
    teacher_kernel = torch.stack(teacher_kernel) if isinstance(teacher_kernel, list) else teacher_kernel
    student_kernel = torch.stack(student_kernel) if isinstance(student_kernel, list) else student_kernel

    if teacher_kernel.size(0) == 1:

        teacher_vec = teacher_kernel.squeeze(0)
        student_vec = student_kernel.squeeze(0)

        common_dim = min(teacher_vec.shape[0], student_vec.shape[0])
        teacher_proj = nn.Linear(teacher_vec.shape[0], common_dim, bias=False).to(teacher_vec.device)
        student_proj = nn.Linear(student_vec.shape[0], common_dim, bias=False).to(student_vec.device)

        teacher_vec = teacher_proj(teacher_vec.unsqueeze(0)).squeeze(0)
        student_vec = student_proj(student_vec.unsqueeze(0)).squeeze(0)

        cos = nn.functional.cosine_similarity(teacher_vec, student_vec, dim=-1)
        loss = (1 - cos) ** 2
        return loss.mean()

    else:
        teacher_matrix = kernel_similarity_matrix(teacher_kernel)
        student_matrix = kernel_similarity_matrix(student_kernel)
        mask = ~torch.eye(teacher_matrix.size(0), dtype=torch.bool, device=teacher_matrix.device)
        teacher_matrix = teacher_matrix[mask]
        student_matrix = student_matrix[mask]

        return mse_loss(teacher_matrix, student_matrix)

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
