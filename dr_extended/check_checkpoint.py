import torch

print("="*60)
print("BASELINE CHECKPOINT")
print("="*60)
baseline_ckpt = "E:/DR_PROJECT/dr_baseline/outputs/checkpoints/best_model.pth"
checkpoint = torch.load(baseline_ckpt, map_location='cpu')

print("\nCORAL layer:")
print(f"  coral.weight: {checkpoint['model_state_dict']['coral.weight'].shape}")
print(f"  coral.bias: {checkpoint['model_state_dict']['coral.bias'].shape}")

print("\nDomain classifier:")
if 'domain_classifier.0.weight' in checkpoint['model_state_dict']:
    print("  Found domain_classifier")
else:
    print("  No domain_classifier")

print("\n" + "="*60)
print("EXTENDED CHECKPOINT")
print("="*60)
extended_ckpt = "E:/DR_PROJECT/dr_extended/outputs/checkpoints/best_model.pth"
checkpoint2 = torch.load(extended_ckpt, map_location='cpu')

print("\nCORAL layer:")
print(f"  coral.weight: {checkpoint2['model_state_dict']['coral.weight'].shape}")
print(f"  coral.bias: {checkpoint2['model_state_dict']['coral.bias'].shape}")

print("\nDomain classifier:")
if 'domain_classifier.0.weight' in checkpoint2['model_state_dict']:
    print("  Found domain_classifier")
    print(f"  domain_classifier.0.weight: {checkpoint2['model_state_dict']['domain_classifier.0.weight'].shape}")
else:
    print("  No domain_classifier")
