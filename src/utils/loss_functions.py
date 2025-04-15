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
    
    # If kernel is a PyTorch tensor, move it to CPU and convert to NumPy
    # if not isinstance(kernel, torch.Tensor):
    #     kernel = kernel.cpu().detach().numpy()  # Move to CPU and detach if needed
    
    #print(type(repr))  # Debugging print
    
    repr = torch.nn.functional.normalize(repr, p=2, dim=1)

    cosine_similarity_matrix = torch.mm(repr, repr.T)

    #print(cosine_similarity_matrix.shape)

    return cosine_similarity_matrix

def kernel_mse_alignment_loss(teacher_kernel, student_kernel):
    """
    Calculates the MSE kernel alignment loss between teacher and student
    """
    #print("zero")
    kernel_similarity_matrix(teacher_kernel)
    #print("zero")
    teacher_matrix = torch.tensor(kernel_similarity_matrix(teacher_kernel))
    #print("zero")
    student_matrix = torch.tensor(kernel_similarity_matrix(student_kernel))
    #print("zero")

    if teacher_matrix.shape != student_matrix.shape:
        teacher_matrix, student_matrix = pad_to_match(teacher_matrix, student_matrix)

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
