import torch
import torch.nn.functional as F

# 这些ID和权重是示例，请根据你的实际情况修改
REASON_START_TOKEN_ID = 151648
REASON_END_TOKEN_ID = 151649
REASON_WEIGHT = 0.8
RESULT_WEIGHT = 1.0
IGNORE_INDEX = -100 # 通常用于padding

def compute_weighted_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    shift_labels: bool = False, # 对于Causal LM需要
    reason_start_token_id: int = REASON_START_TOKEN_ID,
    reason_end_token_id: int = REASON_END_TOKEN_ID,
    reason_weight: float = REASON_WEIGHT,
    result_weight: float = RESULT_WEIGHT,
    ignore_index: int = IGNORE_INDEX
):
    if shift_labels:
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

    batch_size, seq_len, vocab_size = logits.shape


    log_probs = F.log_softmax(logits, dim=-1) # (batch_size, seq_len, vocab_size)

    labels_for_gather = labels.clone()
    labels_for_gather[labels == ignore_index] = 0 # 替换为0，后续会被 mask 掉


    nll_loss_per_token = -log_probs.gather(dim=-1, index=labels_for_gather.unsqueeze(-1)).squeeze(-1)


    token_weights = torch.full_like(labels, fill_value=result_weight, dtype=torch.float, device=labels.device)

    for i in range(batch_size):
        current_labels = labels[i] # (seq_len,)
        try:
       
            start_indices = (current_labels == reason_start_token_id).nonzero(as_tuple=False)
            end_indices = (current_labels == reason_end_token_id).nonzero(as_tuple=False)

            if len(start_indices) > 0 and len(end_indices) > 0:

                start_idx = start_indices[0, 0].item()
                
                # 寻找在 start_idx 之后的第一个 end_idx
                valid_end_idx = -1
                for end_tuple in end_indices:
                    current_end_idx = end_tuple[0].item()
                    if current_end_idx > start_idx:
                        valid_end_idx = current_end_idx
                        break
                
                if valid_end_idx != -1:
                
                    reason_token_indices = slice(start_idx + 1, valid_end_idx)
                    token_weights[i, reason_token_indices] = reason_weight


        except IndexError: # .item() on empty tensor
            pass # 没有找到开始或结束标志，则都按 result_weight 处理

    padding_mask = (labels == ignore_index)
    token_weights.masked_fill_(padding_mask, 0.0)
    nll_loss_per_token.masked_fill_(padding_mask, 0.0) # 确保padding的loss为0

    weighted_nll_loss = nll_loss_per_token * token_weights

    sum_of_weights = token_weights.sum()
    if sum_of_weights > 0:
        final_loss = weighted_nll_loss.sum() / sum_of_weights
    else:
        final_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    return final_loss

class WeightedCELoss:
    def __init__(self, reason_start_token_id, reason_end_token_id, 
                 reason_weight, result_weight, ignore_index):
        self.is_causal_lm = True
        # 或者更简单地，如果你的模型是 AutoModelForCausalLM 类型，可以直接判断
        # self.is_causal_lm = "causal" in model_config.model_type.lower() # 这不完全准确，但有时可用
        # 更可靠的方式是检查模型是否属于 MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        # from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        # self.is_causal_lm = model_config.architectures[0] in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values()
        # (假设 model_config.architectures 存在且包含主模型类名)

        self.reason_start_token_id = reason_start_token_id
        self.reason_end_token_id = reason_end_token_id
        self.reason_weight = reason_weight
        self.result_weight = result_weight
        self.ignore_index = ignore_index
        
        print(f"WeightedCELoss initialized. is_causal_lm: {self.is_causal_lm}")


    def __call__(self, model_outputs, labels, model=None, num_items_in_batch=None): # model和num_items_in_batch可选
        logits = model_outputs.get("logits") if isinstance(model_outputs, dict) else model_outputs[0]
        if logits is None:
            raise ValueError("Model outputs must contain 'logits'.")

        return compute_weighted_cross_entropy_loss(
            logits,
            labels,
            shift_labels=self.is_causal_lm, # 使用初始化时确定的值
            reason_start_token_id=self.reason_start_token_id,
            reason_end_token_id=self.reason_end_token_id,
            reason_weight=self.reason_weight,
            result_weight=self.result_weight,
            ignore_index=self.ignore_index
        )