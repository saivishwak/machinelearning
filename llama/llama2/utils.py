import torch


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000):
    assert head_dim % 2 == 0, "Dim should be even number"

    # Build the theta parameter - theta_i = 10000 ^ (-2 (i-1)/dim) for i = [1, 2, ..... dim/2]
    # Shape - (Head_dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)

    # Construct the positions ("m" parameter)
    # Shape (Seq_len)
    m = torch.arange(seq_len, device=device)

    # Multiply each theta with all possible values of m
    # Shape (Seq_len, Head_dim / 2)
    freqs = torch.outer(m, theta).float()

    # (Seq_len, Head_dim / 2) ->  (Seq_len, Head_dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # (B, Seq_len, H, Head_dim) -> (B, Seq_len, H, Head_dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (Seq_len, Head_Dim/2) -> (1, seq_len, 1, Head_dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs_complex

    # (B, Seq_len, H, Head_dim / 2) -> (B, Seq_len, H, Head_dim / 2, 2)
    x_out = torch.view_as_real(x_rotated)

    # (B, Seq_len, H, Head_dim / 2, 2) -> (B, Seq_len, H, Head_dim)
    x_out = x_out.reshape(*x.shape)

    return x_out.type_as(x).to(device)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        # (B, Seq_len, N_KV_Heads, 1, Head_dim)
        return (
            x[:, :, :, None, :].expand(
                batch_size, seq_len, n_kv_heads, n_rep, head_dim).reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )
