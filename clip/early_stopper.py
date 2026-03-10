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
			print(f"\n{self.__class__.__name__} [initial] Configuration:")
			print(f"  ├─ Patience = {self.patience}")
			print(f"  ├─ MinDelta = {self.min_delta}")
			print(f"  ├─ CumulativeDelta = {self.cumulative_delta}")
			print(f"  ├─ Mode = {self.mode}")
			print(f"  ├─ WindowSize = {self.window_size}")
			print(f"  ├─ MinEpochs = {self.min_epochs}")
			print(f"  ├─ VolatilityThreshold = {self.volatility_threshold}%")
			print(f"  ├─ SlopeThreshold = {self.slope_threshold:.1e}")
			print(f"  ├─ PairwiseImpThreshold = {self.pairwise_imp_threshold}")
			print(f"  ├─ EMA: window = {self.ema_window} recommendation threshold = {self.ema_threshold:.1e}")
			print(f"  ├─ AdaptationFactor = {self.adaptation_factor} [1/factor = relaxation multiplier]")
			print(f"  ├─ MaxPatience = {self.max_patience}")
			if self.min_phases_before_stopping is not None:
				print(f"  ├─ MinPhasesBeforeStopping = {self.min_phases_before_stopping}")
			print(f"  └─ RestoreBestWeights = {self.restore_best_weights}")

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
				
				print(f"\t[Adaptive] EMA unstable → thresholds RELAXED:")
				print(f"\t\tpatience: {self._orig_patience} → {self.effective_patience} (↑{patience_multiplier:.1f}x)")
				print(f"\t\tvolatility_thr: {self._orig_vol_thresh:.1f}% → {self.volatility_threshold:.1f}%")
			elif not is_unstable and self._adaptation_active:
				# Deactivate adaptation: restore original thresholds
				self._adaptation_active = False
				self.effective_patience = self._orig_patience
				self.volatility_threshold = self._orig_vol_thresh
				self.slope_threshold = self._orig_slope_thresh
				print("\t[Adaptive] EMA stable → thresholds restored to original values")
			else:
				# No change
				print("\t[Adaptive] No change in EMA stability")
				pass

		def get_status(self) -> Dict[str, Any]:
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
				# ------------------------------------------------------------------ #
				# 1. Record raw loss and update EMA                                   #
				# ------------------------------------------------------------------ #
				self.value_history.append(current_value)
				self._update_ema(current_value)

				current_ema = self.ema_history[-1]
				phase_info  = f", Phase {current_phase}" if current_phase is not None else ""
				sep         = "─" * 80

				print(f"\n{sep}")
				print(f"EarlyStopping Check — Epoch {epoch+1}{phase_info}")
				print(f"{sep}")
				print(f"  Raw loss        : {current_value:.8f}")
				print(f"  EMA loss        : {current_ema:.8f}")
				print(f"  Best score      : {self.best_score if self.best_score is not None else 'N/A'}")
				print(f"  Best epoch      : {self.best_epoch + 1 if self.best_score is not None else 'N/A'}")
				print(f"  Patience counter: {self.counter} / {self.effective_patience}")
				print(f"  Adaptation      : {'ON' if self._adaptation_active else 'OFF'}")
				print(f"  History length  : {len(self.value_history)} epochs")

				# ------------------------------------------------------------------ #
				# 2. Respect min_epochs — no stopping decisions before warmup ends    #
				# ------------------------------------------------------------------ #
				if epoch < self.min_epochs:
						print(f"\n  [SKIP] Epoch {epoch+1} ≤ min_epochs {self.min_epochs} — skipping all checks")
						# Still track improvement and save checkpoints during warmup
						if self._is_improvement(current_value):
								self._record_improvement(current_value, epoch, model, optimizer, scheduler, checkpoint_path, current_phase)
						else:
								self.counter += 1
								self.improvement_history.append(False)
								print(f"  [WARMUP] No improvement. Counter: {self.counter}/{self.effective_patience}")
						return False

				# ------------------------------------------------------------------ #
				# 3. Improvement check — always runs regardless of window state       #
				# ------------------------------------------------------------------ #
				if self._is_improvement(current_value):
						self._record_improvement(current_value, epoch, model, optimizer, scheduler, checkpoint_path, current_phase)
				else:
						self.counter += 1
						self.improvement_history.append(False)
						delta_from_best = abs(current_value - self.best_score) if self.best_score is not None else float('nan')
						print(f"\n  [NO IMPROVEMENT]")
						print(f"    Delta from best : {delta_from_best:.8f}  (min_delta={self.min_delta})")
						print(f"    Patience counter: {self.counter} / {self.effective_patience}")

				# ------------------------------------------------------------------ #
				# 4. HARD STOP — patience counter exceeded                            #
				# This is the only single-condition stop allowed                      #
				# ------------------------------------------------------------------ #
				if self.counter >= self.effective_patience:
						print(f"\n  [HARD STOP] Patience counter {self.counter} >= {self.effective_patience}")
						if self.restore_best_weights:
								self._restore_best_weights(model)
						return True

				# ------------------------------------------------------------------ #
				# 5. Window checks — require sufficient history                       #
				# ------------------------------------------------------------------ #
				if len(self.value_history) < self.window_size:
						print(f"\n  [SKIP WINDOW] Insufficient history {len(self.value_history)}/{self.window_size}")
						return False

				# ------------------------------------------------------------------ #
				# 6. Compute window statistics on EMA (not raw) for stability         #
				# ------------------------------------------------------------------ #
				raw_window  = self.value_history[-self.window_size:]
				ema_window_vals = self.ema_history[-self.window_size:]

				raw_slope      = compute_slope(raw_window)
				raw_volatility = self._volatility(raw_window)
				ema_slope      = compute_slope(ema_window_vals)
				ema_vol        = self._volatility(ema_window_vals)

				# Pairwise improvement on raw window
				pairwise_diffs = [
						(raw_window[i] - raw_window[i + 1]) * self.sign
						for i in range(len(raw_window) - 1)
				]
				pairwise_improvement = float(np.mean(pairwise_diffs)) if pairwise_diffs else 0.0

				# Relative pairwise improvement (normalised by best score magnitude)
				relative_pairwise = (
						pairwise_improvement / abs(self.best_score)
						if self.best_score is not None and abs(self.best_score) > 1e-10
						else pairwise_improvement
				)

				# Cumulative improvement over the window
				cum_imp_signed = (raw_window[0] - raw_window[-1]) * self.sign
				cum_imp_abs    = abs(cum_imp_signed)

				# EMA trend over recent history
				trend_len    = min(10, len(self.ema_history))
				recent_trend = float(np.mean(np.diff(self.ema_history[-trend_len:]))) if trend_len >= 2 else 0.0

				print(f"\n  {'─'*40}")
				print(f"  WINDOW STATISTICS  (window_size={self.window_size})")
				print(f"  {'─'*40}")
				print(f"  Raw window values   : {[f'{v:.6f}' for v in raw_window]}")
				print(f"  EMA window values   : {[f'{v:.6f}' for v in ema_window_vals]}")
				print(f"  Raw slope           : {raw_slope:+.8f}  (threshold={self.slope_threshold:.1e})")
				print(f"  Raw volatility      : {raw_volatility:.4f}%  (threshold={self.volatility_threshold:.1f}%)")
				print(f"  EMA slope           : {ema_slope:+.8f}  (threshold={self.slope_threshold:.1e})")
				print(f"  EMA volatility      : {ema_vol:.4f}%  (threshold={self.volatility_threshold:.1f}%)")
				print(f"  Pairwise diffs      : {[f'{d:+.6f}' for d in pairwise_diffs]}")
				print(f"  Pairwise improvement: {pairwise_improvement:+.8f}  (absolute)")
				print(f"  Relative pairwise   : {relative_pairwise:+.8f}  (threshold={self.pairwise_imp_threshold})")
				print(f"  Cumulative imp.     : {cum_imp_abs:.8f}  (threshold={self.cumulative_delta})")
				print(f"  EMA recent trend    : {recent_trend:+.8f}  (threshold={self.ema_threshold:.1e}, len={trend_len})")

				# ------------------------------------------------------------------ #
				# 7. Dynamic adaptation based on EMA stability                        #
				# ------------------------------------------------------------------ #
				if len(self.ema_history) >= self.ema_window:
						ema_full_window = self.ema_history[-self.ema_window:]
						ema_full_slope  = compute_slope(ema_full_window)
						ema_full_vol    = self._volatility(ema_full_window)
				else:
						ema_full_slope = 0.0
						ema_full_vol   = 0.0

				print(f"\n  EMA full-window slope : {ema_full_slope:+.8f}")
				print(f"  EMA full-window vol   : {ema_full_vol:.4f}%")
				self._apply_dynamic_adaptation(ema_full_slope, ema_full_vol)

				# ------------------------------------------------------------------ #
				# 8. Evaluate SOFT stopping signals — require >= 2 to agree           #
				# Using EMA-based volatility (not raw) for stability                  #
				# ------------------------------------------------------------------ #
				close_to_best = (
						abs(current_value - self.best_score) < self.min_delta
						if self.best_score is not None else False
				)

				soft_signals: List[str] = []

				# Signal 1 — EMA volatility (replaces raw volatility)
				if ema_vol >= self.volatility_threshold:
						soft_signals.append(
								f"High EMA volatility ({ema_vol:.4f}%) >= {self.volatility_threshold:.1f}%"
						)

				# Signal 2 — worsening slope (EMA-based)
				ema_worsening = (
						(self.mode == "min" and ema_slope > self.slope_threshold) or
						(self.mode == "max" and ema_slope < -self.slope_threshold)
				)
				if ema_worsening:
						soft_signals.append(
								f"Worsening EMA slope ({ema_slope:+.8f}) vs threshold ({self.slope_threshold:.1e})"
						)

				# Signal 3 — low relative pairwise improvement
				if relative_pairwise < self.pairwise_imp_threshold and not close_to_best:
						soft_signals.append(
								f"Low relative pairwise improvement ({relative_pairwise:+.8f}) "
								f"< {self.pairwise_imp_threshold} [close_to_best={close_to_best}]"
						)

				# Signal 4 — low cumulative improvement over window
				if cum_imp_abs < self.cumulative_delta:
						soft_signals.append(
								f"Low cumulative improvement ({cum_imp_abs:.8f}) < {self.cumulative_delta}"
						)

				# Signal 5 — EMA trend worsening
				if recent_trend > self.ema_threshold:
						soft_signals.append(
								f"EMA trend worsening ({recent_trend:+.8f}) > {self.ema_threshold:.1e} over {trend_len} epochs"
						)

				# ------------------------------------------------------------------ #
				# 9. Print soft signal evaluation                                     #
				# ------------------------------------------------------------------ #
				print(f"\n  {'─'*60}")
				print(f"  SOFT SIGNAL EVALUATION  (require >= 2 to trigger stop)")
				print(f"  {'─'*60}")

				all_signal_names = [
						("EMA volatility",           ema_vol >= self.volatility_threshold),
						("Worsening EMA slope",      ema_worsening),
						("Low relative pairwise",    relative_pairwise < self.pairwise_imp_threshold and not close_to_best),
						("Low cumulative imp.",      cum_imp_abs < self.cumulative_delta),
						("EMA trend worsening",      recent_trend > self.ema_threshold),
				]
				for signal_name, fired in all_signal_names:
						status = "✗ FIRED" if fired else "✓ OK   "
						print(f"    [{status}] {signal_name}")

				print(f"\n  Soft signals fired: {len(soft_signals)} / {len(all_signal_names)}")
				if soft_signals:
						for s in soft_signals:
								print(f"    • {s}")
				else:
						print(f"    (none)")

				# ------------------------------------------------------------------ #
				# 10. Final stopping decision                                         #
				# ------------------------------------------------------------------ #
				should_stop_flag = len(soft_signals) >= 2

				print(f"\nFINAL DECISION:")

				if should_stop_flag:
						# Phase constraint check (only relevant for progressive fine-tuning)
						if self.min_phases_before_stopping is not None:
								phase_ok = (current_phase is None) or (current_phase >= self.min_phases_before_stopping)
								if not phase_ok:
										print(f"  [DEFER] Soft signals >= 2 but waiting for phase >= {self.min_phases_before_stopping}")
										print(f"  => CONTINUE TRAINING")
										should_stop_flag = False
								else:
										print(f"  [SOFT STOP] {len(soft_signals)} signals agreed:")
										for s in soft_signals:
												print(f"    • {s}")
						else:
								print(f"  [SOFT STOP] {len(soft_signals)} signals agreed:")
								for s in soft_signals:
										print(f"    • {s}")
				else:
						print(f"\t=> NO stopping conditions met — CONTINUE TRAINING")

				print(f"{sep}\n")

				# ------------------------------------------------------------------ #
				# 11. Restore best weights if stopping                                #
				# ------------------------------------------------------------------ #
				if should_stop_flag and self.restore_best_weights:
						self._restore_best_weights(model)

				return should_stop_flag

		def _record_improvement(
				self,
				current_value: float,
				epoch: int,
				model: torch.nn.Module,
				optimizer: torch.optim.Optimizer,
				scheduler,
				checkpoint_path: str,
				current_phase: Optional[int],
		) -> None:
				"""Centralised improvement recording — called from should_stop and warmup block."""
				old_best = f"{self.best_score:.8f}" if self.best_score is not None else "N/A"
				improvement = (
						abs(self.best_score - current_value)
						if self.best_score is not None else float('nan')
				)
				self.best_score  = current_value
				self.best_epoch  = epoch
				self.counter     = 0
				self.improvement_history.append(True)

				print(f"\n  [NEW BEST]")
				print(f"    Previous best : {old_best}")
				print(f"    New best      : {current_value:.8f}")
				print(f"    Improvement   : {improvement:.8f}")
				print(f"    Epoch         : {epoch + 1}")

				if self.restore_best_weights:
						self.best_weights = {
								k: v.clone().cpu().detach()
								for k, v in model.state_dict().items()
						}
						print(f"    Best weights  : stored in memory")

				self._save_checkpoint(checkpoint_path, model, optimizer, scheduler, current_phase)

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
						print(f"\tSaved checkpoint: {path}")
				except Exception as e:
						print(f"  Warning: Failed to save checkpoint - {e}")

		def _restore_best_weights(self, model):
			if self.best_weights is not None:
				target_device = next(model.parameters()).device
				model.load_state_dict({k: v.to(target_device) for k, v in self.best_weights.items()})
				print(f"  Restored best weights from epoch {self.best_epoch+1} (loss: {self.best_score})")
			else:
				print("  Warning: No best weights stored - cannot restore")
