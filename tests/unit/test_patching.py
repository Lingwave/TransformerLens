import pytest
import torch

from transformer_lens import patching, HookedTransformer, ActivationCache


MODEL = "solu-1l"

model = HookedTransformer.from_pretrained(MODEL)

def test_make_df_from_ranges():
    column_max_ranges = [10, 20, 40]
    column_names = ["layer", "post", "value"]
    result = patching.make_df_from_ranges(column_max_ranges=column_max_ranges, column_names=column_names)
    print(result.shape)
    assert result.shape == torch.Size([8000, 3])
    
def test_layer_pos_patch_setter_fails_with_invalid_index():
    with pytest.raises(Exception) as e_info:
        corrupted_activation = torch.rand(3, 2)
        index = [0]
        clean_activation = torch.rand(3, 2)
        result = patching.layer_pos_patch_setter(corrupted_activation, index, clean_activation)

def test_layer_pos_patch_setter():
    corrupted_activation = torch.rand(3, 2)
    index = [0, 1]
    clean_activation = torch.rand(3, 2)

    original_value = corrupted_activation[:, 1, ...].clone()
    expected_replacements = clean_activation[:, 1, ...].clone()

    result = patching.layer_pos_patch_setter(corrupted_activation, index, clean_activation)
    
    new_value = result[:, 1, ...]

    assert result.shape == torch.Size([3,2])
    assert torch.equal(new_value, expected_replacements)
    assert not torch.equal(new_value, original_value)
    
def test_layer_pos_head_vector_patch_setter_fails_with_invalid_index():
    with pytest.raises(Exception) as e_info:
        corrupted_activation = torch.rand(3, 2)
        index = [0]
        clean_activation = torch.rand(3, 2)
        result = patching.layer_pos_head_vector_patch_setter(corrupted_activation, index, clean_activation)
    
def test_layer_pos_head_vector_patch_setter():
    pass
    
def test_layer_head_vector_patch_setter_fails_with_invalid_index():
    with pytest.raises(Exception) as e_info:
        corrupted_activation = torch.rand(3, 2)
        index = [0]
        clean_activation = torch.rand(3, 2)
        result = patching.layer_head_vector_patch_setter(corrupted_activation, index, clean_activation)

def test_layer_head_vector_patch_setter():
    pass

def test_layer_head_pattern_patch_setter_fails_with_invalid_index():
    with pytest.raises(Exception) as e_info:
        corrupted_activation = torch.rand(3, 2)
        index = [0]
        clean_activation = torch.rand(3, 2)
        result = patching.layer_head_pattern_patch_setter(corrupted_activation, index, clean_activation)

def test_layer_head_pattern_patch_setter():
    pass

def test_layer_head_pos_pattern_patch_setter_fails_with_invalid_index():
    with pytest.raises(Exception) as e_info:
        corrupted_activation = torch.rand(3, 2)
        index = [0]
        clean_activation = torch.rand(3, 2)
        result = patching.layer_head_pos_pattern_patch_setter(corrupted_activation, index, clean_activation)

def test_layer_head_pos_pattern_patch_setter():
    pass

def test_layer_head_dest_src_pos_pattern_patch_setter_fails_with_invalid_index():
    with pytest.raises(Exception) as e_info:
        corrupted_activation = torch.rand(3, 2)
        index = [0]
        clean_activation = torch.rand(3, 2)
        result = patching.layer_head_dest_src_pos_pattern_patch_setter(corrupted_activation, index, clean_activation)

def test_layer_head_dest_src_pos_pattern_patch_setter():
    pass

def test_get_act_patch_attn_head_all_pos_every():
    pass

def test_get_act_patch_attn_head_by_pos_every():
    pass

def test_get_act_patch_block_every():
    pass
    
def test_generic_activation_patch():
    corrupted_activation = torch.rand(3, 2)
    cache_dict = {"test" : torch.rand(2,3)}
    activationCache = ActivationCache(cache_dict= cache_dict, model=model)
    