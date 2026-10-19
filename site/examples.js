"use strict";
// One teaching shape generates both complete programs and their plan previews.
window.Examples = (() => {
  const shape={T:16,D:256,F:256,K:2,E_local:4,EP:2,L:1,N:1,deviceFactor:1,slotFactor:1};
  const bf16=`import os

import torch
import torch.distributed as dist

import dist_moe


device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group("nccl")
ep_group = dist.group.WORLD

T, D, F, K, E_local = ${shape.T}, ${shape.D}, ${shape.F}, ${shape.K}, ${shape.E_local}
rank = dist.get_rank(ep_group)
ep_size = dist.get_world_size(ep_group)
E = E_local * ep_size

config = dist_moe.Config(
    max_num_local_input_tokens=T,
    hidden_dim=D,
    intermediate_dim=F,
    top_k=K,
    num_experts=E,
    max_moe_layers_per_activation_slot=${shape.L},
    device_scratch_capacity_factor=${shape.deviceFactor.toFixed(1)},
    activation_slot_capacity_factor=${shape.slotFactor.toFixed(1)},
    num_activation_slots=${shape.N},
)
plan = dist_moe.plan_memory(config, ep_size=ep_size)
print(plan.explain())

context = dist_moe.create_context(
    group=ep_group,
    config=config,
    device=device,
)

try:
    x_TD = torch.randn(
        T,
        D,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    global_token_ids_T = rank * T + torch.arange(T, device=device)
    topk_expert_ids_TK = (
        global_token_ids_T[:, None] * K
        + torch.arange(K, device=device)[None, :]
    ) % E
    topk_scores_TK = torch.full(
        (T, K),
        1.0 / K,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    w13_E2FD = torch.randn(
        E_local,
        2,
        F,
        D,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    w2_EDF = torch.randn(
        E_local,
        D,
        F,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )

    output_TD = dist_moe.routed_experts(
        x_TD, topk_expert_ids_TK, topk_scores_TK,
        w13_E2FD, w2_EDF, context,
    )
    output_TD.float().sum().backward()
    assert output_TD.shape == x_TD.shape
    assert x_TD.grad is not None
finally:
    context.close()
    dist.destroy_process_group()`;
  const policy='policy = dist_moe.BlockScaledConfig(\n    format=dist_moe.BlockScaledFormat.MXFP8_E4M3,\n    pipeline="staged",  # or "mega"\n)\n';
  const mxfp8=bf16.replace('config = dist_moe.Config(',policy+'config = dist_moe.Config(')
    .replace(`    num_activation_slots=${shape.N},`,`    num_activation_slots=${shape.N},\n    block_scaled=policy,`)
    .replace('    output_TD = dist_moe.routed_experts(',
      '    w13_EFD = w13_E2FD.flatten(1, 2)\n    prepared_w13 = dist_moe.prepare_block_scaled_weight(\n        w13_EFD, policy,\n    )\n    prepared_w2 = dist_moe.prepare_block_scaled_weight(\n        w2_EDF, policy,\n    )\n    output_TD = dist_moe.routed_experts(')
    .replace('        w13_E2FD, w2_EDF, context,','        prepared_w13, prepared_w2, context,');
  function config(precision){return {...shape,E:shape.E_local*shape.EP,totalFactor:shape.deviceFactor,vmm:false,precision,pipeline:'staged',activation_slot_bytes:null,activation_slot_capacity_factor:shape.slotFactor};}
  return {shape,bf16,mxfp8,config};
})();
