from utils import *

def simple_moving_average(data: np.ndarray, window: int) -> np.ndarray:
		"""SMA – thin wrapper around pandas for speed."""
		return pd.Series(data).rolling(window=window, min_periods=1).mean().values

def exponential_moving_average(
		data: np.ndarray, window: int, alpha: Optional[float] = None
) -> np.ndarray:
		"""EMA – same implementation that LossAnalyzer used."""
		if alpha is None:
				alpha = 2.0 / (window + 1)          # standard EMA smoothing factor
		ema = np.empty_like(data)
		ema[0] = data[0]
		for i in range(1, len(data)):
				ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
		return ema

class EarlyStopping:
		"""
		Enhanced early stopping with EMA smoothing and adaptive thresholds.
		Works safely with Full, LoRA, and Progressive fine-tuning methods.
		"""

		def __init__(
				self,
				patience: int = 5,
				min_delta: float = 1e-3,
				cumulative_delta: float = 0.01,
				window_size: int = 5,
				mode: str = "min",
				min_epochs: int = 5,
				restore_best_weights: bool = True,
				volatility_threshold: float = 10.0,
				slope_threshold: float = 0.001,  # Small positive default instead of 0.0
				pairwise_imp_threshold: float = 5e-3,
				min_phases_before_stopping: Optional[int] = None,
				ema_window: int = 10,
				ema_threshold: float = 1e-3,
				adaptation_factor: float = 0.5,
				max_patience: int = 30,  # Hard ceiling to prevent runaway patience
		):
				# Core parameters
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
				self.adaptation_factor = adaptation_factor
				self.ema_window = ema_window
				self.ema_threshold = ema_threshold
				self.max_patience = max_patience

				# Internal setup
				self.sign = 1 if mode == "min" else -1
				self.ema_alpha = 2.0 / (ema_window + 1)

				self.reset()
				self._log_config()

		def _log_config(self) -> None:
			config_summary = f"""
			Patience = {self.patience}
			MinDelta = {self.min_delta}
			CumulativeDelta = {self.cumulative_delta}
			Mode = {self.mode}
			WindowSize = {self.window_size}
			MinEpochs = {self.min_epochs}
			VolatilityThreshold = {self.volatility_threshold}%
			SlopeThreshold = {self.slope_threshold:.1e}
			PairwiseImpThreshold = {self.pairwise_imp_threshold}
			EMA: window = {self.ema_window} recommendation threshold = {self.ema_threshold:.1e}
			AdaptationFactor = {self.adaptation_factor} [1/factor = relaxation multiplier]
			MaxPatience = {self.max_patience}
			RestoreBestWeights = {self.restore_best_weights}
			"""
			if self.min_phases_before_stopping is not None:
				config_summary += f"MinPhasesBeforeStopping = {self.min_phases_before_stopping}"
			print("=" * 100)
			print(f"{self.__class__.__name__} [initial] Configuration")
			print(config_summary)
			print("=" * 100)

		def reset(self) -> None:
				"""Reset state for new training or phase"""
				print(f">> Resetting {self.__class__.__name__} state")
				
				# Best model tracking
				self.best_score: Optional[float] = None
				self.best_weights: Optional[Dict[str, torch.Tensor]] = None
				self.best_epoch: int = 0
				
				# Counters
				self.counter: int = 0
				self.effective_patience: int = self.patience
				
				# Histories
				self.value_history: List[float] = []
				self.ema_history: List[float] = []
				self.improvement_history: List[bool] = []
				
				# Store original thresholds for restoration
				self._orig_patience = self.patience
				self._orig_vol_thresh = self.volatility_threshold
				self._orig_slope_thresh = self.slope_threshold
				
				# Adaptation state
				self._adaptation_active = False

		def _update_ema(self, raw_val: float) -> None:
				"""Update exponential moving average"""
				if not self.ema_history:
						self.ema_history.append(raw_val)
				else:
						prev = self.ema_history[-1]
						new_ema = self.ema_alpha * raw_val + (1 - self.ema_alpha) * prev
						self.ema_history.append(new_ema)

		@staticmethod
		def _volatility(window: List[float]) -> float:
				"""Coefficient of variation as percentage"""
				if len(window) < 2:
						return 0.0
				mean_val = np.mean(window)
				std_val = np.std(window)
				return (std_val / abs(mean_val)) * 100.0 if mean_val != 0 else 0.0

		def _is_improvement(self, cur_val: float) -> bool:
				"""Check if current value represents improvement"""
				if self.best_score is None:
						return True
				improvement = (self.best_score - cur_val) * self.sign
				return improvement > self.min_delta

		def _apply_dynamic_adaptation(self, ema_slope: float, ema_vol: float) -> None:
				"""
				Apply immediate adaptation based on EMA stability:
				- Unstable EMA: Relax thresholds (more patience)  
				- Stable EMA: Restore original thresholds
				"""
				is_unstable = (ema_slope > self.slope_threshold) or (ema_vol > self.volatility_threshold)
				
				if is_unstable and not self._adaptation_active:
						# Activate adaptation: relax thresholds
						self._adaptation_active = True
						patience_multiplier = 1.0 / self.adaptation_factor
						self.effective_patience = min(self.max_patience, int(self._orig_patience * patience_multiplier))
						self.volatility_threshold = self._orig_vol_thresh / self.adaptation_factor
						self.slope_threshold = self._orig_slope_thresh / self.adaptation_factor
						
						print(f"  [Adaptive] EMA unstable → thresholds RELAXED:")
						print(f"    patience: {self._orig_patience} → {self.effective_patience} (↑{patience_multiplier:.1f}x)")
						print(f"    volatility_thr: {self._orig_vol_thresh:.1f}% → {self.volatility_threshold:.1f}%")
						
				elif not is_unstable and self._adaptation_active:
						# Deactivate adaptation: restore original thresholds
						self._adaptation_active = False
						self.effective_patience = self._orig_patience
						self.volatility_threshold = self._orig_vol_thresh
						self.slope_threshold = self._orig_slope_thresh
						print("  [Adaptive] EMA stable → thresholds restored to original values")

		def get_status(self) -> Dict[str, Any]:
				"""Enhanced status information"""
				status = {
						"best_score": self.best_score,
						"best_epoch": self.best_epoch + 1 if self.best_score is not None else None,
						"patience_counter": self.counter,
						"effective_patience": self.effective_patience,
						"adaptation_active": self._adaptation_active,
						"value_history_len": len(self.value_history),
						"ema_history_len": len(self.ema_history),
				}
				
				# Add window statistics if available
				if len(self.value_history) >= self.window_size:
						win = self.value_history[-self.window_size:]
						status["volatility_window"] = self._volatility(win)
						status["slope_window"] = compute_slope(win)
				
				return status

		def get_best_score(self) -> Optional[float]:
				return self.best_score

		def get_best_epoch(self) -> int:
				return self.best_epoch

		def should_stop(
				self,
				current_value: float,
				model: torch.nn.Module,
				epoch: int,
				optimizer: torch.optim.Optimizer,
				scheduler,
				checkpoint_path: str,
				current_phase: Optional[int] = None,
		) -> bool:
				# 1. Record raw loss & EMA
				self.value_history.append(current_value)
				self._update_ema(current_value)
				
				current_ema = self.ema_history[-1]
				phase_info = f", Phase {current_phase}" if current_phase is not None else ""
				
				print(f"\n{self.__class__.__name__} Check (Epoch {epoch+1}{phase_info})")
				print(f"\tRaw loss: {current_value} | EMA loss: {current_ema}")
				print(f"\tCounter: {self.counter}/{self.effective_patience} | Adaptation: {'ON' if self._adaptation_active else 'OFF'}")
				
				# 2. Respect min_epochs
				if epoch < self.min_epochs:
					print(f"\tSkipping check (epoch {epoch+1} ≤ min_epochs {self.min_epochs})")
					return False

				# 3. Handle improvement
				if self._is_improvement(current_value):
					old_best = f"{self.best_score:.6f}" if self.best_score is not None else "N/A"
					print(f"  >> NEW BEST! {old_best} → {current_value:.6f}")
					
					self.best_score = current_value
					self.best_epoch = epoch
					self.counter = 0
					self.improvement_history.append(True)
					
					# Store best weights
					if self.restore_best_weights:
							self.best_weights = {k: v.clone().cpu().detach() for k, v in model.state_dict().items()}
					
					# Save checkpoint with better error handling
					self._save_checkpoint(checkpoint_path, model, optimizer, scheduler, current_phase)
				else:
					self.counter += 1
					self.improvement_history.append(False)
					best_str = f"{self.best_score:.6f}" if self.best_score is not None else "N/A"
					print(f"  No improvement. Best: {best_str}")

				# 4. Check if we have enough history for window-based analysis
				if len(self.value_history) < self.window_size:
					print(f"  Insufficient history ({len(self.value_history)}/{self.window_size}) for window checks")
					if self.counter >= self.effective_patience:
						# Only check phase constraints if min_phases_before_stopping is set
						if self.min_phases_before_stopping is not None:
							phase_ok = (current_phase is None) or (current_phase >= self.min_phases_before_stopping)
							if phase_ok:
								print("  Patience exceeded → early stop")
								return True
							else:
								print(f"  Patience exceeded but waiting for phase >= {self.min_phases_before_stopping}")
								return False
						else:
							print("  Patience exceeded → early stop")
							return True
					return False

				# 5. Compute window statistics
				raw_window = self.value_history[-self.window_size:]
				raw_slope = compute_slope(raw_window)
				raw_volatility = self._volatility(raw_window)
				
				# Pairwise improvement analysis
				pairwise_diffs = [
					(raw_window[i] - raw_window[i + 1]) * self.sign 
					for i in range(len(raw_window) - 1)
				]
				pairwise_improvement = np.mean(pairwise_diffs) if pairwise_diffs else 0.0
				
				# Cumulative improvement
				cum_imp_signed = (raw_window[0] - raw_window[-1]) * self.sign
				cum_imp_abs = abs(cum_imp_signed)

				# 6. EMA-based analysis and dynamic adaptation
				if len(self.ema_history) >= self.ema_window:
					ema_window_vals = self.ema_history[-self.ema_window:]
					ema_slope = compute_slope(ema_window_vals)
					ema_vol = self._volatility(ema_window_vals)
				else:
					ema_slope = 0.0
					ema_vol = 0.0

				self._apply_dynamic_adaptation(ema_slope, ema_vol)

				# 7. Assemble stopping reasons
				stop_reasons: List[str] = []
				
				if self.counter >= self.effective_patience:
					stop_reasons.append(f"Patience ({self.counter}/{self.effective_patience})")
				
				if raw_volatility >= self.volatility_threshold:
					stop_reasons.append(f"High volatility ({raw_volatility:.2f}%)")
				
				worsening = (self.mode == "min" and raw_slope > self.slope_threshold) or (self.mode == "max" and raw_slope < self.slope_threshold)
				if worsening:
					stop_reasons.append(f"Worsening slope ({raw_slope})")
				
				close_to_best = (
					abs(current_value - self.best_score) < self.min_delta 
					if self.best_score is not None else False
				)
				if pairwise_improvement < self.pairwise_imp_threshold and not close_to_best:
					stop_reasons.append(f"Low pairwise improvement ({pairwise_improvement})")
				
				if cum_imp_abs < self.cumulative_delta:
					stop_reasons.append(f"Low cumulative improvement ({cum_imp_abs})")
				
				# EMA trend analysis (only add if clearly worsening)
				if len(self.ema_history) >= 3:
					trend_len = min(10, len(self.ema_history))
					recent_trend = np.mean(np.diff(self.ema_history[-trend_len:])) if trend_len >= 2 else 0.0
					if recent_trend > self.ema_threshold:
						stop_reasons.append("EMA trend worsening")

				# 8. Final decision with phase constraints
				should_stop = bool(stop_reasons)
				if should_stop:
					# Only apply phase constraints if min_phases_before_stopping is set (progressive fine-tuning)
					if self.min_phases_before_stopping is not None:
						phase_ok = (current_phase is None) or (current_phase >= self.min_phases_before_stopping)
						if not phase_ok:
							print(f"  Stopping criteria met ({', '.join(stop_reasons)}) but waiting for phase >= {self.min_phases_before_stopping}")
							should_stop = False
						else:
							print(f"  EARLY STOPPING TRIGGERED:")
							for reason in stop_reasons:
								print(f"    • {reason}")
					else:
						print(f"  EARLY STOPPING TRIGGERED:")
						for reason in stop_reasons:
							print(f"    • {reason}")
				else:
					print("  No stopping conditions met")

				# 9. Restore best weights if stopping
				if should_stop and self.restore_best_weights:
					self._restore_best_weights(model)

				return should_stop

		def _save_checkpoint(self, path: str, model, optimizer, scheduler, current_phase):
				"""Save checkpoint with enhanced error handling"""
				try:
						checkpoint = {
								"epoch": self.best_epoch,
								"model_state_dict": self.best_weights if self.best_weights is not None else model.state_dict(),
								"optimizer_state_dict": optimizer.state_dict(),
								"scheduler_state_dict": scheduler.state_dict(),
								"best_val_loss": self.best_score,
								"effective_patience": self.effective_patience,
								"adaptation_active": self._adaptation_active,
						}
						if current_phase is not None:
								checkpoint["phase"] = current_phase
						
						torch.save(checkpoint, path)
						print(f"  Saved checkpoint: {path}")
				except Exception as e:
						print(f"  Warning: Failed to save checkpoint - {e}")

		def _restore_best_weights(self, model):
				"""Restore best weights with better messaging"""
				if self.best_weights is not None:
						target_device = next(model.parameters()).device
						model.load_state_dict({k: v.to(target_device) for k, v in self.best_weights.items()})
						print(f"  Restored best weights from epoch {self.best_epoch+1} (loss: {self.best_score:.6f})")
				else:
						print("  Warning: No best weights stored - cannot restore")

class EarlyStoppingOld:
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

		slope = compute_slope(window=last_window)
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
			status["slope_window"] = compute_slope(window=last_window)
		else:
			status["volatility_window"] = None
			status["slope_window"] = None
		return status

	def get_best_score(self) -> Optional[float]:
		return self.best_score

	def get_best_epoch(self) -> int:
		return self.best_epoch