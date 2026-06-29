import os
import argparse
from stage5_regime_conditioned_training import regime_conditioned_finetune

def parse_args():
	p = argparse.ArgumentParser(description="Stage 5: Regime-Conditioned Training")
	p.add_argument("--metadata", '-csv',required=True,  help="Path to metadata.csv")
	p.add_argument("--clip_model", '-cm', default="ViT-B/32")
	p.add_argument("--peft_method", '-peft', default="lora",  choices=["lora","lora+","dora","rslora","ia3","vera","probe","adapter","full"])
	p.add_argument("--epochs",'-e', type=int, default=30)
	p.add_argument("--batch_size", '-bs', type=int,   default=256)
	p.add_argument("--num_workers", '-nw', type=int,   default=4)
	p.add_argument("--learning_rate", '-lr', type=float, default=1e-4)
	p.add_argument("--pw_mode",       default="sqrt", choices=["log","sqrt","linear"])
	p.add_argument("--pw_max_cap",    type=float, default=50.0)
	p.add_argument("--patience",      type=int,   default=7)
	p.add_argument("--resume_ckpt",   default=None,   help="Resume training from a checkpoint")
	p.add_argument("--seed",          type=int,   default=42)
	p.add_argument("--verbose", "-v", action='store_true', help="Print verbose diagnostics")

	return p.parse_args()

if __name__ == "__main__":
	args = parse_args()
	DATASET_DIRECTORY = os.path.dirname(args.metadata)
	print(DATASET_DIRECTORY)

	OUTPUTs_DIRECTORY = os.path.join(DATASET_DIRECTORY, "outputs")
	os.makedirs(OUTPUTs_DIRECTORY, exist_ok=True)
	print(OUTPUTs_DIRECTORY)

	CHECKPOINTs_DIRECTORY = os.path.join(OUTPUTs_DIRECTORY, "checkpoints")
	os.makedirs(CHECKPOINTs_DIRECTORY, exist_ok=True)
	print(CHECKPOINTs_DIRECTORY)

	metadata_fpath = os.path.basename(args.metadata)
	supervision_fpath = os.path.join(
		OUTPUTs_DIRECTORY, 
		metadata_fpath.replace(".csv", "_mlm_cot_modality_conflict_audit_auditable_supervision_matrix.parquet")
	)
	
	print(supervision_fpath)
	assert os.path.exists(supervision_fpath), f"Supervision matrix not found at {supervision_fpath}"

	regime_conditioned_finetune(
		metadata_fpth    = args.metadata,
		supervision_fpth = supervision_fpath,
		checkpoints_dir  = CHECKPOINTs_DIRECTORY,
		clip_model_name  = args.clip_model,
		peft_method      = args.peft_method,
		num_epochs       = args.epochs,
		batch_size       = args.batch_size,
		num_workers      = args.num_workers,
		learning_rate    = args.learning_rate,
		pw_mode          = args.pw_mode,
		pw_max_cap       = args.pw_max_cap,
		patience         = args.patience,
		resume_ckpt      = args.resume_ckpt,
		seed             = args.seed,
		verbose          = args.verbose,
	)