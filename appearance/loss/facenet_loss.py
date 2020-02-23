import torch

def triplet_loss(anchor, positive, negative, alpha=0.3):
    """Calculate the triplet loss according to the FaceNet paper
    
    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.
  
    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    pos_dist = torch.pow(anchor-positive, 2).sum(dim=1, keepdim=True) #(N, 1)
    neg_dist = torch.pow(anchor-negative, 2).sum(dim=1, keepdim=True) #(N, 1)
    basic_loss = pos_dist - neg_dist + alpha
    loss = torch.mean(torch.clamp(basic_loss, min=0.0), dim=0)
      
    return loss


if __name__ == '__main__':
    anchor = torch.randn(2, 512)
    positive = torch.randn(2, 512)
    negative = torch.randn(2, 512)
    print(triplet_loss(anchor, positive, negative))


