from utils import *

class EarlyStopping:
	def __init__(
			self,
			patience: int = 5,
			min_delta: float = 1e-3,
			cumulative_delta: float = 0.01,
			window_size: int = 5,
			mode: str = 'min',
			min_epochs: int = 5,
			restore_best_weights: bool = True,
			volatility_threshold: float = 10.0,
			slope_threshold: float = 0.0,
			pairwise_imp_threshold: float = 5e-3,
			min_phases_before_stopping: int = 3,
		):

		self.patience = patience
		self.min_delta = min_delta
		self.cumulative_delta = cumulative_delta
		self.window_size = window_size
		self.mode = mode
		self.min_epochs = min_epochs
		self.restore_best_weights = restore_best_weights
		self.volatility_threshold = volatility_threshold
		self.slope_threshold = slope_threshold
		self.pairwise_imp_threshold = pairwise_imp_threshold
		self.min_phases_before_stopping = min_phases_before_stopping
		self.sign = 1 if mode == 'min' else -1
		print("="*100)
		print(
			f"EarlyStopping [initial] Configuration:\n"
			f"\tPatience={patience}\n"
			f"\tMinDelta={min_delta}\n"
			f"\tCumulativeDelta={cumulative_delta}\n"
			f"\tWindowSize={window_size}\n"
			f"\tMinEpochs={min_epochs}\n"
			f"\tMinPhases={min_phases_before_stopping} (only for progressive finetuning)\n"
			f"\tVolatilityThreshold={volatility_threshold}\n"
			f"\tSlopeThreshold={slope_threshold}\n"
			f"\tPairwiseImpThreshold={pairwise_imp_threshold}\n"
			f"\tRestoreBestWeights={restore_best_weights}"
		)
		self.reset()
		print("="*100)

	def reset(self):
		print(">> Resetting EarlyStopping state, Essential for starting fresh or resetting between training phases")
		self.best_score = None
		self.best_weights = None
		self.counter = 0
		self.stopped_epoch = 0
		self.best_epoch = 0
		self.value_history = []
		self.improvement_history = []
		self.current_phase = 0
		self.model_improved_this_epoch = False

	def compute_volatility(self, window: List[float]) -> float:
		if not window or len(window) < 2:
			return 0.0
		mean_val = np.mean(window)
		std_val = np.std(window)
		return (std_val / abs(mean_val)) * 100 if mean_val != 0 else 0.0

	def is_improvement(self, current_value: float) -> bool:
		if self.best_score is None:
			return True
		improvement = (self.best_score - current_value) * self.sign
		return improvement > self.min_delta

	def should_stop(
			self,
			current_value: float,
			model: torch.nn.Module,
			optimizer: torch.optim.Optimizer,
			scheduler,
			epoch: int,
			checkpoint_path: str,
			current_phase: Optional[int] = None,
		) -> bool:

		self.model_improved_this_epoch = False
		self.value_history.append(current_value)
		phase_info = f", Phase {current_phase}" if current_phase is not None else ""
		print(f"\n--- EarlyStopping Check (Epoch {epoch+1}{phase_info}) ---")
		print(f"Current validation loss: {current_value}")

		if epoch < self.min_epochs:
			print(f"Skipping early stopping (epoch {epoch+1} <= min_epochs {self.min_epochs})")
			return False

		if self.is_improvement(current_value):
			print(
				f"\t>>>> New Best Model Found! "
				f"Loss improved from {self.best_score if self.best_score is not None else 'N/A'} to {current_value}"
			)
			self.best_score = current_value
			self.best_epoch = epoch
			self.counter = 0
			self.improvement_history.append(True)
			self.model_improved_this_epoch = True

			if self.restore_best_weights:
				self.best_weights = {k: v.clone().cpu().detach() for k, v in model.state_dict().items()}
			
			print(f"Saving new best model checkpoint (from epoch {self.best_epoch + 1}) to {checkpoint_path}")
			checkpoint = {
				"epoch": self.best_epoch,
				"model_state_dict": self.best_weights if self.best_weights is not None else model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"scheduler_state_dict": scheduler.state_dict(),
				"best_val_loss": self.best_score,
			}
			if current_phase is not None:
				checkpoint["phase"] = current_phase
			try:
				torch.save(checkpoint, checkpoint_path)
			except Exception as e:
				print(f"Warning: Failed to save checkpoint to {checkpoint_path}: {e}")
		else:
			self.counter += 1
			self.improvement_history.append(False)
			print(
				f"\tNO improvement! Best: {self.best_score} "
				f"Patience: {self.counter}/{self.patience}"
			)

		if len(self.value_history) < self.window_size:
			print(f"\tNot enough history ({len(self.value_history)} < {self.window_size}) for window-based checks.")
			if self.counter >= self.patience:
				phase_constraint_met = (current_phase is None) or (current_phase >= self.min_phases_before_stopping)
				if phase_constraint_met:
					print(f"EARLY STOPPING TRIGGERED: Patience ({self.counter}/{self.patience}) exceeded.")
					return True
			return False

		last_window = self.value_history[-self.window_size:]
		print(f"\tWindow ({self.window_size} epochs): {last_window}")

		slope = compute_slope(last_window)
		print(f"\tSlope over {self.window_size} windows: {slope} (Threshold > {self.slope_threshold})")
		
		volatility = self.compute_volatility(last_window)
		print(f"\tVolatility over {self.window_size} windows: {volatility:.2f}% (Threshold >= {self.volatility_threshold}%)")
		
		pairwise_diffs = [(last_window[i] - last_window[i+1]) * self.sign for i in range(len(last_window)-1)]
		pairwise_imp_avg = np.mean(pairwise_diffs) if pairwise_diffs else 0.0
		print(f"\tAvg Pairwise Improvement: {pairwise_imp_avg} (Threshold < {self.pairwise_imp_threshold})")
		
		close_to_best = abs(current_value - self.best_score) < self.min_delta if self.best_score is not None else False
		print(f"\tClose to best score ({self.best_score}): {close_to_best}")
		
		window_start_value = self.value_history[-self.window_size]
		window_end_value = self.value_history[-1]
		cumulative_improvement_signed = (window_start_value - window_end_value) * self.sign
		cumulative_improvement_abs = abs(cumulative_improvement_signed)
		print(f"\tCumulative Improvement: {cumulative_improvement_signed} (Threshold < {self.cumulative_delta})")
		
		stop_reason = []
		if self.counter >= self.patience:
			stop_reason.append(f"Patience ({self.counter}/{self.patience})")
		if volatility >= self.volatility_threshold:
			stop_reason.append(f"High volatility ({volatility:.2f}%)")
		is_worsening = (self.mode == 'min' and slope > self.slope_threshold) or \
						 (self.mode == 'max' and slope < self.slope_threshold)
		if is_worsening:
			stop_reason.append(f"Worsening slope ({slope:.5f})")
		if pairwise_imp_avg < self.pairwise_imp_threshold and not close_to_best:
			stop_reason.append(f"Low pairwise improvement ({pairwise_imp_avg:.5f}) & not close to best")
		if cumulative_improvement_abs < self.cumulative_delta:
			stop_reason.append(f"Low cumulative improvement ({cumulative_improvement_abs:.5f})")

		should_trigger_stop = bool(stop_reason)
		should_really_stop = False

		if should_trigger_stop:
			reason_str = ', '.join(stop_reason)
			phase_constraint_met = (current_phase is None) or (current_phase >= self.min_phases_before_stopping)
			if phase_constraint_met:
				print(f"<!> EARLY STOPPING TRIGGERED:\n\t{reason_str}")
				should_really_stop = True
			else:
				print(f"\tEarly stopping condition triggered ({reason_str}), but delaying stop (Phase {current_phase} < {self.min_phases_before_stopping})")
		else:
			print("\tNo stopping conditions met.")

		if should_really_stop and self.restore_best_weights:
			if self.best_weights is not None:
				target_device = next(model.parameters()).device
				print(f"Restoring model weights from best epoch {self.best_epoch + 1} (score: {self.best_score})")
				model.load_state_dict({k: v.to(target_device) for k, v in self.best_weights.items()})
			else:
				print("Warning: restore_best_weights is True, but no best weights were saved.")
		
		return should_really_stop

	def get_status(self) -> Dict[str, Any]:
		status = {
			"best_score": self.best_score,
			"best_epoch": self.best_epoch + 1 if self.best_score is not None else 0,
			f"patience_counter(out of {self.patience})": self.counter,
			"value_history_len": len(self.value_history)
		}
		if len(self.value_history) >= self.window_size:
			last_window = self.value_history[-self.window_size:]
			status["volatility_window"] = self.compute_volatility(last_window)
			status["slope_window"] = compute_slope(last_window)
		else:
			status["volatility_window"] = None
			status["slope_window"] = None
		return status

	def get_best_score(self) -> Optional[float]:
		return self.best_score

	def get_best_epoch(self) -> int:
		return self.best_epoch