"""Parameter-free multiplicative Q/K score node used by bilinear attention."""
import torch


def dot_score(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    if query.ndim != 3 or key.shape != query.shape or not query.is_floating_point():
        raise ValueError("query/key ports must be matching floating [batch,position,head_dim] tensors")
    return torch.einsum("bqd,bkd->bqk", query, key) / query.shape[-1]


def execute(query1: torch.Tensor, key1: torch.Tensor, query2: torch.Tensor, key2: torch.Tensor) -> torch.Tensor:
    if query2.shape != query1.shape or key2.shape != key1.shape:
        raise ValueError("both Q/K branches must share port shapes")
    return dot_score(query1, key1) * dot_score(query2, key2)


def compose_scores(score1_parts, score2_parts):
    """Expand a product of additive branch scores into pair interactions."""
    if not score1_parts or not score2_parts:
        raise ValueError("composition requires nonempty score-part lists")
    return sum(left * right for left in score1_parts for right in score2_parts)
