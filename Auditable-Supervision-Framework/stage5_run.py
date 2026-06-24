# run_stage5.py
import argparse
from stage5_regime_conditioned_training import regime_conditioned_finetune

def parse_args():
	p = argparse.ArgumentParser(description="Stage 5: Regime-Conditioned Training")
	p.add_argument("--metadata", '-csv',required=True,  help="Path to metadata.csv")
	p.add_argument("--supervision", '-parquet',   required=True,  help="Path to auditable_supervision_matrix.parquet")
	p.add_argument("--output_dir",    default="./stage5_outputs")
	p.add_argument("--clip_model",    default="ViT-B/32")
	p.add_argument("--peft_method",   default="lora",  choices=["lora","lora+","dora","rslora","ia3","vera","probe","adapter","full"])
	p.add_argument("--epochs",        type=int,   default=30)
	p.add_argument("--batch_size",    type=int,   default=128)
	p.add_argument("--num_workers",   type=int,   default=4)
	p.add_argument("--lr",            type=float, default=1e-4)
	p.add_argument("--pw_mode",       default="sqrt", choices=["log","sqrt","linear"])
	p.add_argument("--pw_max_cap",    type=float, default=50.0)
	p.add_argument("--patience",      type=int,   default=7)
	p.add_argument("--resume_ckpt",   default=None,   help="Path to checkpoint to resume from")
	p.add_argument("--seed",          type=int,   default=42)
	return p.parse_args()

if __name__ == "__main__":
	args = parse_args()
	regime_conditioned_finetune(
		metadata_fpth    = args.metadata,
		supervision_fpth = args.supervision,
		output_dir       = args.output_dir,
		clip_model_name  = args.clip_model,
		peft_method      = args.peft_method,
		num_epochs       = args.epochs,
		batch_size       = args.batch_size,
		num_workers      = args.num_workers,
		learning_rate    = args.lr,
		pw_mode          = args.pw_mode,
		pw_max_cap       = args.pw_max_cap,
		patience         = args.patience,
		resume_ckpt      = args.resume_ckpt,
		seed             = args.seed,
	)