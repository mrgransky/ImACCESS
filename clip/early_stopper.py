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
				slope_threshold: float = 0.0,
				pairwise_imp_threshold: float = 5e-3,
				min_phases_before_stopping: Optional[int] = None,
				ema_window: int = 10,
				ema_threshold: float = 1e-3,
				adaptation_factor: float = 0.5,
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
				self.adaptation_factor = adaptation_factor
				self.ema_window = ema_window
				self.ema_threshold = ema_threshold   # used for the high‑level recommendation

				self.sign = 1 if mode == "min" else -1

				# ----- EMA specifics -----------------------------------------
				self.ema_alpha = 2.0 / (ema_window + 1)
				self.ema_history: List[float] = []   # EMA of validation loss

				self.reset()
				self._log_config()

		def _log_config(self) -> None:
			config_summary = f"""
			Patience = {self.patience}
			MinDelta = {self.min_delta}
			CumulativeDelta = {self.cumulative_delta}
			WindowSize = {self.window_size}
			Mode = {self.mode}
			MinEpochs = {self.min_epochs}
			VolatilityThreshold = {self.volatility_threshold}
			SlopeThreshold = {self.slope_threshold}
			PairwiseImpThreshold = {self.pairwise_imp_threshold}
			EMA window = {self.ema_window}
			EMA recommendation threshold = {self.ema_threshold}
			AdaptationFactor = {self.adaptation_factor} [1/factor = relaxation multiplier]
			RestoreBestWeights = {self.restore_best_weights}
			"""
			if self.min_phases_before_stopping is not None:
				config_summary += f"MinPhasesBeforeStopping = {self.min_phases_before_stopping}"
			print("=" * 100)
			print(f"{self.__class__.__name__} [initial] Configuration")
			print(config_summary)
			print("=" * 100)

		def reset(self) -> None:
				print(f">> Resetting {self.__class__.__name__} state")
				self.best_score: Optional[float] = None
				self.best_weights: Optional[Dict[str, torch.Tensor]] = None
				self.best_epoch: int = 0
				self.counter: int = 0          # patience counter (no‑improve epochs)

				self.value_history: List[float] = []   # raw validation loss
				self.ema_history = []                 # EMA of validation loss
				self.improvement_history: List[bool] = []
				self.current_phase: int = 0

				self.train_loss_history: List[float] = []   # optional – over‑fit gap

				# restored after every dynamic adaptation step
				self._orig_patience = self.patience
				self._orig_vol_thresh = self.volatility_threshold
				self._orig_slope_thresh = self.slope_threshold

				self._unstable_last = False

		def _update_ema(self, raw_val: float) -> None:
				if not self.ema_history:
						self.ema_history.append(raw_val)
				else:
						prev = self.ema_history[-1]
						self.ema_history.append(self.ema_alpha * raw_val + (1 - self.ema_alpha) * prev)

		@staticmethod
		def _volatility(window: List[float]) -> float:
			if len(window) < 2:
				print(f"<!> Not enough data ({len(window)} < 2) to compute volatility.")
				return 0.0
			mean_val = np.mean(window)
			std_val = np.std(window)
			return (std_val / abs(mean_val)) * 100.0 if mean_val != 0 else 0.0

		def _is_improvement(self, cur_val: float) -> bool:
			if self.best_score is None:
				return True
			improvement = (self.best_score - cur_val) * self.sign
			return improvement > self.min_delta

		def _apply_dynamic_adaptation(self, ema_slope: float, ema_vol: float) -> None:
				"""
				Dynamically adapt stopping thresholds based on EMA stability:
				- When EMA is noisy/unstable: RELAX thresholds (more patience)
				- When EMA is stable: Use original thresholds
				
				This prevents premature stopping during inherently noisy periods
				like progressive fine-tuning phase transitions.
				"""
				is_unstable = (ema_slope > self.slope_threshold) or (ema_vol > self.volatility_threshold)
				
				if is_unstable:
						# Remember we're in an unstable period
						self._unstable_last = True  # Also rename this flag
						
						# RELAX thresholds to accommodate noise
						patience_multiplier = 1.0 / self.adaptation_factor  # e.g., 0.5 -> 2.0x patience
						self.patience = int(self._orig_patience * patience_multiplier)
						self.volatility_threshold = self._orig_vol_thresh / self.adaptation_factor
						self.slope_threshold = self._orig_slope_thresh / self.adaptation_factor
						
						print("\t[Adaptive] EMA unstable → thresholds RELAXED for stability:")
						print(
								f"\t   patience → {self.patience} (↑{patience_multiplier:.1f}x)   "
								f"volatility_thr → {self.volatility_threshold:.2f}%   "
								f"slope_thr → {self.slope_threshold:.5e}"
						)
				else:
						# Restore original thresholds once stability returns
						if self._unstable_last:
								self.patience = self._orig_patience
								self.volatility_threshold = self._orig_vol_thresh
								self.slope_threshold = self._orig_slope_thresh
								self._unstable_last = False
								print("\t[Adaptive] EMA stable → thresholds restored to original values.")

		def get_status(self) -> Dict[str, Any]:
			status = {
				"best_score": self.best_score,
				"best_epoch": self.best_epoch + 1 if self.best_score is not None else None,
				"patience_counter": self.counter,
				"value_history_len": len(self.value_history),
				"ema_history_len": len(self.ema_history),
			}
			if len(self.value_history) >= self.window_size:
				win = self.value_history[-self.window_size :]
				status["volatility_window"] = self._volatility(win)
				status["slope_window"] = compute_slope(win)
			else:
				status["volatility_window"] = None
				status["slope_window"] = None
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
			
			# 1 Record raw loss & EMA
			self.value_history.append(current_value)
			self._update_ema(current_value)
			phase_info = f", Phase {current_phase}" if current_phase is not None else ""
			print(f"\n--- {self.__class__.__name__} Check (Epoch {epoch+1}{phase_info}) ---")
			print(f"Raw validation loss: {current_value}")
			print(f"EMA (window={self.ema_window}) loss: {self.ema_history[-1]}")
			
			# 2 Warm-up & Respect min_epochs
			if epoch < self.min_epochs:
				print(f"Skipping early‑stop (epoch {epoch+1} ≤ min_epochs {self.min_epochs})")
				return False

			# 3 Update best checkpoint if we improved (raw loss)
			if self._is_improvement(current_value):
				best_str = f"{self.best_score}" if self.best_score is not None else "N/A"
				print(f"\t>> NEW BEST MODEL! loss improved from {best_str} to {current_value}")
				self.best_score = current_value
				self.best_epoch = epoch
				self.counter = 0
				self.improvement_history.append(True)
				if self.restore_best_weights:
					self.best_weights = {k: v.clone().cpu().detach() for k, v in model.state_dict().items()}
				ckpt = {
					"epoch": self.best_epoch,
					"model_state_dict": self.best_weights if self.best_weights is not None else model.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"scheduler_state_dict": scheduler.state_dict(),
					"best_val_loss": self.best_score,
				}
				if current_phase is not None: ckpt["phase"] = current_phase
				try:
					torch.save(ckpt, checkpoint_path)
					print(f"Saved new best checkpoint → {checkpoint_path}")
				except Exception as exc:
					print(f"<!> Failed to save checkpoint: {exc}")
			else:
				# No improvement → increase patience counter
				self.counter += 1
				self.improvement_history.append(False)
				best_str = f"{self.best_score:.6f}" if self.best_score is not None else "N/A"
				print(
					f"\tNo improvement. Best: {best_str} "
					f"Patience: {self.counter}/{self.patience}"
				)

			# 4 Check if we have enough history for window-based analysis
			if len(self.value_history) < self.window_size:
				print(f"\tInsufficient history ({len(self.value_history)}/{self.window_size}) for window‑based checks")
				if self.counter >= self.patience:
					# Only check phase constraints if min_phases_before_stopping is set:
					if self.min_phases_before_stopping is not None:
						phase_ok =(current_phase is None) or (current_phase >= self.min_phases_before_stopping)
						if phase_ok:
							print(">> Patience exceeded → early stop.")
							return True
						else:
							print(f"Patience exceeded but waiting for phase >= {self.min_phases_before_stopping}")
							return False
					else:
						print(">> Patience exceeded → early stop.")
						return True
				return False
			raw_window = self.value_history[-self.window_size :]
			raw_slope = compute_slope(raw_window)
			raw_volatility = self._volatility(raw_window)
			# Pairwise improvement (sign‑aware)
			pairwise_diffs = [
				(raw_window[i] - raw_window[i + 1]) * self.sign
				for i in range(len(raw_window) - 1)
			]
			pairwise_improvement = np.mean(pairwise_diffs) if pairwise_diffs else 0.0
			# Cumulative improvement across the whole raw window
			cum_imp_signed = (raw_window[0] - raw_window[-1]) * self.sign
			cum_imp_abs = abs(cum_imp_signed)

			# 5 EMA‑based trend signals (used for dynamic adaptation)			
			if len(self.ema_history) >= self.ema_window:
				ema_window_vals = self.ema_history[-self.ema_window :]
				ema_slope = compute_slope(ema_window_vals)
				ema_vol = self._volatility(ema_window_vals)
			else:
				ema_slope = 0.0
				ema_vol = 0.0

			# 6 Apply the dynamic‑adaptation policy
			self._apply_dynamic_adaptation(ema_slope, ema_vol)

			# 7 Assemble stop reasons (raw‑loss criteria + EMA recommendation)
			stop_reasons: List[str] = []
			if self.counter >= self.patience:
				stop_reasons.append(f"Patience ({self.counter}/{self.patience})")
			if raw_volatility >= self.volatility_threshold:
				stop_reasons.append(f"High volatility ({raw_volatility:.2f}%)")
			worsening = (self.mode == "min" and raw_slope > self.slope_threshold) or (
				self.mode == "max" and raw_slope < self.slope_threshold
			)
			if worsening:
				stop_reasons.append(f"Worsening slope ({raw_slope:.5e})")
			close_to_best = (
				abs(current_value - self.best_score) < self.min_delta
				if self.best_score is not None
				else False
			)
			if pairwise_improvement < self.pairwise_imp_threshold and not close_to_best:
				stop_reasons.append(f"Low pairwise improvement ({pairwise_improvement:.5e}) & not close to best")
			if cum_imp_abs < self.cumulative_delta:
				stop_reasons.append(f"Low cumulative improvement ({cum_imp_abs:.5e})")
			
			# 8 EMA‑based high‑level recommendation (mirrors LossAnalyzer)
			# Use a safe window length – at most the EMA length, at least 2 points
			trend_len = min(10, len(self.ema_history))
			recent_trend = np.mean(np.diff(self.ema_history[-trend_len:])) if trend_len >= 2 else 0.0
			if recent_trend > self.ema_threshold:
				stop_reasons.append("EMA trend ↑ (recommend STOP)")
			elif recent_trend > -self.ema_threshold:
				stop_reasons.append("EMA trend ≈0 (recommend CAUTION)")
			
			# 9 Final decision with phase constraints (if set)
			should_stop = bool(stop_reasons)
			if should_stop:
				# Only apply phase constraints if min_phases_before_stopping is set (progressive fine-tuning)
				if self.min_phases_before_stopping is not None:
					phase_ok = (current_phase is None) or (current_phase >= self.min_phases_before_stopping)
					if not phase_ok:
						print(
							f"\tStopping criteria met ({', '.join(stop_reasons)}) "
							f"but waiting for phase >= {self.min_phases_before_stopping}"
						)
						should_stop = False
					else:
						print("\n<!> EARLY STOPPING TRIGGERED:")
						for r in stop_reasons:
							print(f"\t • {r}")
				else:
					print("\n<!> EARLY STOPPING TRIGGERED:")
					for r in stop_reasons:
						print(f"\t • {r}")
			else:
				print("\tNo stopping condition satisfied this epoch.")
			
			if should_stop and self.restore_best_weights:
				if self.best_weights is not None:
					target_device = next(model.parameters()).device
					model.load_state_dict(
						{k: v.to(target_device) for k, v in self.best_weights.items()}
					)
					print(
						f">> Restored best weights from epoch {self.best_epoch+1} "
						f"(score={self.best_score:.6f})"
					)
				else:
					print("<!> No best weights stored – cannot restore.")
			return should_stop

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