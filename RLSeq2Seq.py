import torch


def rl_decode_seq2seq(
    model,
    src,
    max_len,
    sos_idx,
    eos_idx,
    pad_idx,
    sample=True,
    temperature=1.0,
    no_repeat=True,
    forbid_src_tokens=True,
):
    """
    src: [src_len, B], batch_first=False
    returns:
        generated: [max_len, B]
        log_probs_sum: [B]
    """

    device = src.device
    src_len, batch_size = src.shape
    vocab_size = model.decoder.output_dim

    hidden = model.encoder(src)

    input_token = torch.full(
        (batch_size,),
        fill_value=sos_idx,
        dtype=torch.long,
        device=device,
    )

    generated_tokens = []
    log_probs_list = []

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    used_mask = torch.zeros(
        batch_size,
        vocab_size,
        dtype=torch.bool,
        device=device,
    )

    with torch.no_grad():
        used_mask[:, sos_idx] = True
        used_mask[:, pad_idx] = True

        if forbid_src_tokens:
            src_b = src.transpose(0, 1)  # [B, src_len]
            src_b = src_b.clamp(min=0, max=vocab_size - 1)
            used_mask.scatter_(1, src_b, True)

    for _ in range(max_len):
        logits, hidden = model.decoder(input_token, hidden)  # [B, V]

        # Important: clone logits and mask.
        # Do not let autograd depend on a mask that will be modified later.
        logits = logits.clone()

        if no_repeat:
            mask_for_logits = used_mask.clone()
            logits = logits.masked_fill(mask_for_logits, -1e9)

        # finished examples produce pad with zero log-prob contribution
        if finished.any():
            finished_mask = finished.clone()
            logits[finished_mask, :] = -1e9
            logits[finished_mask, pad_idx] = 0.0

        probs = torch.softmax(logits / temperature, dim=-1)

        if sample:
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_token = probs.argmax(dim=-1)

        step_log_prob = torch.log(
            probs.gather(1, next_token.unsqueeze(1)).squeeze(1) + 1e-12
        )

        step_log_prob = step_log_prob.masked_fill(finished.clone(), 0.0)

        generated_tokens.append(next_token)
        log_probs_list.append(step_log_prob)

        with torch.no_grad():
            if no_repeat:
                used_mask = used_mask.clone()
                used_mask.scatter_(1, next_token.unsqueeze(1), True)

            finished = finished.clone()
            finished |= next_token.eq(eos_idx)

        input_token = next_token.detach()

    generated = torch.stack(generated_tokens, dim=0)              # [max_len, B]
    log_probs_sum = torch.stack(log_probs_list, dim=0).sum(dim=0) # [B]

    return generated, log_probs_sum


def seq_set_f1_reward(pred_seq, gold_seq, ignore_ids):
    """
    pred_seq: [T, B]
    gold_seq: [T, B]
    returns: [B]
    """

    rewards = []
    batch_size = pred_seq.shape[1]

    for b in range(batch_size):
        pred = set(int(x) for x in pred_seq[:, b].detach().cpu().tolist())
        gold = set(int(x) for x in gold_seq[:, b].detach().cpu().tolist())

        pred = pred - ignore_ids
        gold = gold - ignore_ids

        if len(pred) == 0 and len(gold) == 0:
            rewards.append(1.0)
            continue

        if len(pred) == 0 or len(gold) == 0:
            rewards.append(0.0)
            continue

        tp = len(pred & gold)
        precision = tp / len(pred)
        recall = tp / len(gold)

        if precision + recall == 0:
            rewards.append(0.0)
        else:
            rewards.append(2 * precision * recall / (precision + recall))

    return torch.tensor(rewards, dtype=torch.float, device=pred_seq.device)


def train_seq_rl(
    model,
    data_loader,
    optimizer,
    criterion,
    clip,
    device,
    pad_idx,
    rl_weight=1.0,
    ce_weight=0.0,
    temperature=1.0,
):
    model.train()

    epoch_loss = 0.0
    epoch_reward = 0.0
    sos_idx, eos_idx, pad_idx = pad_idx

    ignore_ids = {pad_idx, sos_idx, eos_idx}

    for batch in data_loader:
        src = batch["input_ids"].to(device)   # [src_len, B]
        trg = batch["pos_ids"].to(device)     # [trg_len, B]

        optimizer.zero_grad()

        max_len = trg.shape[0] - 1

        sampled_seq, sampled_log_probs = rl_decode_seq2seq(
            model=model,
            src=src,
            max_len=max_len,
            sos_idx=sos_idx,
            eos_idx=eos_idx,
            pad_idx=pad_idx,
            sample=True,
            temperature=temperature,
            no_repeat=True,
            forbid_src_tokens=True,
        )

        with torch.no_grad():
            greedy_seq, _ = rl_decode_seq2seq(
                model=model,
                src=src,
                max_len=max_len,
                sos_idx=sos_idx,
                eos_idx=eos_idx,
                pad_idx=pad_idx,
                sample=False,
                temperature=1.0,
                no_repeat=True,
                forbid_src_tokens=True,
            )

        gold_seq = trg[1:]

        sampled_reward = seq_set_f1_reward(
            pred_seq=sampled_seq,
            gold_seq=gold_seq,
            ignore_ids=ignore_ids,
        )

        greedy_reward = seq_set_f1_reward(
            pred_seq=greedy_seq,
            gold_seq=gold_seq,
            ignore_ids=ignore_ids,
        )

        advantage = (sampled_reward - greedy_reward).detach()

        rl_loss = -(advantage * sampled_log_probs).mean()

        if ce_weight > 0.0:
            output = model(src, trg, model.args.teacher_forcing_ratio)
            output_dim = output.shape[-1]

            ce_loss = criterion(
                output[1:].reshape(-1, output_dim),
                trg[1:].reshape(-1),
            )

            loss = rl_weight * rl_loss + ce_weight * ce_loss
        else:
            loss = rl_weight * rl_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        epoch_reward += sampled_reward.mean().item()

    return epoch_loss / len(data_loader)


