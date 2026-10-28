"""
Unit tests for model.py. Everything runs on CPU with a tiny GPT in a few seconds.
$ python -m pytest
"""
import math

import pytest
import torch

from model import GPTConfig, GPT

def tiny_config(**overrides):
    args = dict(block_size=16, vocab_size=65, n_layer=2, n_head=2, n_embd=32, dropout=0.0, bias=True)
    args.update(overrides)
    return GPTConfig(**args)

@pytest.fixture
def model():
    torch.manual_seed(0)
    return GPT(tiny_config()).eval()

def random_batch(config, b=4, t=None):
    t = t or config.block_size
    x = torch.randint(config.vocab_size, (b, t))
    y = torch.randint(config.vocab_size, (b, t))
    return x, y

def test_forward_shapes(model):
    x, y = random_batch(model.config)
    logits, loss = model(x, y)
    assert logits.shape == (4, 16, 65)
    assert loss.dim() == 0
    # without targets only the last position goes through the lm_head
    logits, loss = model(x)
    assert logits.shape == (4, 1, 65)
    assert loss is None

def test_initial_loss_is_close_to_uniform(model):
    # with the small init, the untrained model should predict roughly uniformly over the vocab
    x, y = random_batch(model.config, b=8)
    _, loss = model(x, y)
    assert abs(loss.item() - math.log(65)) < 0.15

def test_ignore_index_skips_masked_targets(model):
    x, y = random_batch(model.config)
    masked = y.clone()
    masked[:, :8] = -1
    logits, loss = model(x, masked)
    expected = torch.nn.functional.cross_entropy(logits[:, 8:].reshape(-1, 65), y[:, 8:].reshape(-1))
    assert torch.allclose(loss, expected, atol=1e-6)

def test_sequence_longer_than_block_size_is_rejected(model):
    x = torch.zeros((1, 17), dtype=torch.long)
    with pytest.raises(AssertionError):
        model(x)

def test_weights_are_tied(model):
    assert model.lm_head.weight is model.transformer.wte.weight

def test_num_params_excludes_position_embeddings(model):
    total = sum(p.numel() for p in model.parameters())
    assert model.get_num_params(non_embedding=False) == total
    assert model.get_num_params() == total - 16 * 32

def test_backward_reaches_every_parameter(model):
    model.train()
    x, y = random_batch(model.config)
    _, loss = model(x, y)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name

def test_manual_attention_matches_flash(model):
    # force the slow path on a copy of the same weights and compare
    x, _ = random_batch(model.config)
    flash_logits, _ = model(x)
    T = model.config.block_size
    for block in model.transformer.h:
        block.attn.flash = False
        block.attn.register_buffer("bias", torch.tril(torch.ones(T, T)).view(1, 1, T, T))
    manual_logits, _ = model(x)
    assert torch.allclose(flash_logits, manual_logits, atol=1e-5)

def test_attention_is_causal(model):
    # changing a later token must not change the predictions for earlier positions
    x, y = random_batch(model.config, b=1)
    logits_a, _ = model(x, y)
    x2 = x.clone()
    x2[0, -1] = (x2[0, -1] + 1) % 65
    logits_b, _ = model(x2, y)
    assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-6)
    assert not torch.allclose(logits_a[:, -1], logits_b[:, -1])

def test_generate_appends_tokens(model):
    idx = torch.zeros((2, 3), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=5)
    assert out.shape == (2, 8)
    assert torch.equal(out[:, :3], idx)
    assert out.min() >= 0 and out.max() < 65

def test_generate_crops_context_longer_than_block_size(model):
    idx = torch.zeros((1, 20), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=4)
    assert out.shape == (1, 24)

def test_generate_top_k_1_is_greedy(model):
    idx = torch.zeros((1, 4), dtype=torch.long)
    torch.manual_seed(1)
    a = model.generate(idx, max_new_tokens=6, top_k=1)
    torch.manual_seed(2)
    b = model.generate(idx, max_new_tokens=6, top_k=1)
    assert torch.equal(a, b)
    # greedy decoding picks the argmax at every step
    seq = idx
    for _ in range(6):
        logits, _ = model(seq)
        seq = torch.cat((seq, logits[:, -1, :].argmax(dim=-1, keepdim=True)), dim=1)
    assert torch.equal(a, seq)

def test_crop_block_size(model):
    model.crop_block_size(8)
    assert model.config.block_size == 8
    assert model.transformer.wpe.weight.shape == (8, 32)
    logits, _ = model(torch.zeros((1, 8), dtype=torch.long))
    assert logits.shape == (1, 1, 65)
    with pytest.raises(AssertionError):
        model(torch.zeros((1, 9), dtype=torch.long))

def test_configure_optimizers_groups(model):
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=1e-3, betas=(0.9, 0.95), device_type='cpu')
    decay, nodecay = optimizer.param_groups
    assert decay['weight_decay'] == 0.1 and nodecay['weight_decay'] == 0.0
    # weight matrices and embeddings decay, biases and layernorm weights don't
    assert all(p.dim() >= 2 for p in decay['params'])
    assert all(p.dim() < 2 for p in nodecay['params'])
    n_grouped = sum(p.numel() for g in optimizer.param_groups for p in g['params'])
    assert n_grouped == sum(p.numel() for p in model.parameters())

def test_no_bias_config():
    m = GPT(tiny_config(bias=False))
    assert all(not name.endswith('.bias') for name, _ in m.named_parameters())
    x, y = random_batch(m.config)
    _, loss = m(x, y)
    assert torch.isfinite(loss)

def test_overfits_a_single_batch():
    # a few hundred steps on one batch should drive the loss far below the uniform baseline
    torch.manual_seed(0)
    m = GPT(tiny_config())
    x, y = random_batch(m.config, b=2)
    optimizer = m.configure_optimizers(weight_decay=0.0, learning_rate=3e-3, betas=(0.9, 0.95), device_type='cpu')
    for _ in range(200):
        _, loss = m(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.5
