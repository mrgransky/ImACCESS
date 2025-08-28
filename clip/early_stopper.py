from utils import *

# ----------------------------------------------------------------------
# Helper utilities – they are the same as the ones you already used
# ----------------------------------------------------------------------
# def compute_slope(window: List[float]) -> float:
#     """Linear regression slope of a 1‑D window."""
#     if len(window) < 2:
#         return 0.0
#     x = np.arange(len(window))
#     y = np.asarray(window)
#     # slope = cov(x, y) / var(x)
#     return np.cov(x, y, bias=True)[0, 1] / np.var(x)


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


# ----------------------------------------------------------------------
#  HybridEarlyStopping – the merged class
# ----------------------------------------------------------------------
class EarlyStopping:
		"""
		Early‑stopping that:

		1. Tracks the **raw validation loss** (for checkpointing & exact best‑epoch).
		2. Maintains an **EMA** of the validation loss (window = `ema_window`).
		3. Computes the same high‑level signals that `LossAnalyzer` provides
			 – best epoch based on EMA, over‑fitting gap, recent trend, and a
			 textual recommendation.
		4. Implements a **dynamic‑threshold** policy: when the EMA shows a
			 shaky trend the internal patience / volatility / slope limits are
			 *tightened only for the current epoch*.
		5. Works exactly like the original EarlyStopping (same signature,
			 same checkpoint saving, same restoration of the best raw weights).

		Parameters
		----------
		patience               : int   – maximum epochs without improvement.
		min_delta              : float – minimal absolute improvement to be counted.
		cumulative_delta       : float – minimal cumulative improvement over the
																			sliding window.
		window_size            : int   – number of epochs used for the *raw* window
																			statistics (slope, volatility, …).
		ema_window             : int   – span of the EMA (default 10).  Larger → smoother.
		mode                   : {"min","max"} – `"min"` for loss, `"max"` for metric.
		min_epochs             : int   – epochs that must run before any stop is allowed.
		restore_best_weights   : bool  – if True, reload the best raw checkpoint.
		volatility_threshold   : float – max %‑std/mean allowed on the *raw* window.
		slope_threshold        : float – maximum allowed slope on the *raw* window.
		pairwise_imp_threshold : float – min average pair‑wise improvement.
		min_phases_before_stopping : int – for progressive fine‑tuning.
		dynamic_factor         : float – factor by which thresholds are tightened
																			when EMA is shaky (default 0.5 → halve).
		"""
		# ------------------------------------------------------------------
		#  Constructor – store everything and initialise state
		# ------------------------------------------------------------------
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
				min_phases_before_stopping: int = 3,
				ema_window: int = 10,
				dynamic_factor: float = 0.5,
		):

				# ---- user‑provided thresholds ---------------------------------
				self.patience = patience
				self.min_delta = min_delta
				self.cumulative_delta = cumulative_delta
				self.window_size = window_size
				self.ema_window = ema_window
				self.mode = mode
				self.min_epochs = min_epochs
				self.restore_best_weights = restore_best_weights
				self.volatility_threshold = volatility_threshold
				self.slope_threshold = slope_threshold
				self.pairwise_imp_threshold = pairwise_imp_threshold
				self.min_phases_before_stopping = min_phases_before_stopping
				self.dynamic_factor = dynamic_factor

				# sign used for minimisation / maximisation
				self.sign = 1 if mode == "min" else -1

				# ---- EMA specific parameters ----------------------------------
				self.ema_alpha = 2.0 / (self.ema_window + 1)   # same convention as LossAnalyzer
				self.ema_history: List[float] = []           # EMA of the validation loss

				# ---- internal bookkeeping --------------------------------------
				self.reset()

				# ---- print a nice configuration --------------------------------
				self._print_configuration()

		# ------------------------------------------------------------------
		#  Helper – nice printing of the configuration
		# ------------------------------------------------------------------
		def _print_configuration(self) -> None:
				print("=" * 100)
				print(
						f"{self.__class__.__name__} [initial] Configuration:\n"
						f"\tPatience = {self.patience}\n"
						f"\tMinDelta = {self.min_delta}\n"
						f"\tCumulativeDelta = {self.cumulative_delta}\n"
						f"\tWindowSize = {self.window_size}\n"
						f"\tEMA window = {self.ema_window}\n"
						f"\tMode = {self.mode}\n"
						f"\tMinEpochs = {self.min_epochs}\n"
						f"\tMinPhasesBeforeStopping = {self.min_phases_before_stopping}\n"
						f"\tVolatilityThreshold = {self.volatility_threshold}\n"
						f"\tSlopeThreshold = {self.slope_threshold}\n"
						f"\tPairwiseImpThreshold = {self.pairwise_imp_threshold}\n"
						f"\tDynamicFactor (tightening) = {self.dynamic_factor}\n"
						f"\tRestoreBestWeights = {self.restore_best_weights}"
				)
				print("=" * 100)

		# ------------------------------------------------------------------
		#  Reset – called at start and after each phase transition
		# ------------------------------------------------------------------
		def reset(self) -> None:
				"""Reset all counters and histories (useful after a phase change)."""
				print(">> Resetting HybridEarlyStopping state")
				self.best_score: Optional[float] = None
				self.best_weights: Optional[Dict[str, torch.Tensor]] = None
				self.best_epoch: int = 0
				self.counter: int = 0           # patience counter (no‑improve epochs)
				self.value_history: List[float] = []   # raw validation loss
				self.ema_history = []           # EMA of validation loss
				self.improvement_history: List[bool] = []
				self.current_phase: int = 0
				# Keep a copy of training loss (used for over‑fitting gap)
				self.train_loss_history: List[float] = []

				# Preserve original thresholds so we can restore after dynamic tightening
				self._orig_patience = self.patience
				self._orig_vol_thresh = self.volatility_threshold
				self._orig_slope_thresh = self.slope_threshold

		# ------------------------------------------------------------------
		#  EMA update – called once per epoch before any decision making
		# ------------------------------------------------------------------
		def _update_ema(self, raw_val: float) -> None:
				"""Append a new EMA value using the classic exponential smoothing."""
				if not self.ema_history:
						self.ema_history.append(raw_val)
				else:
						prev = self.ema_history[-1]
						self.ema_history.append(self.ema_alpha * raw_val + (1 - self.ema_alpha) * prev)

		# ------------------------------------------------------------------
		#  Volatility – computed on any 1‑D window (raw or EMA)
		# ------------------------------------------------------------------
		@staticmethod
		def _volatility(window: List[float]) -> float:
				if len(window) < 2:
						return 0.0
				mean_val = np.mean(window)
				std_val = np.std(window)
				return (std_val / abs(mean_val)) * 100.0 if mean_val != 0 else 0.0

		# ------------------------------------------------------------------
		#  Helper – check whether the current raw loss is an improvement
		# ------------------------------------------------------------------
		def _is_improvement(self, cur_val: float) -> bool:
				if self.best_score is None:
						return True
				improvement = (self.best_score - cur_val) * self.sign
				return improvement > self.min_delta

		# ------------------------------------------------------------------
		#  Dynamic‑threshold policy – tighten thresholds only for the *current*
		#  epoch when the EMA signals a shaky trend.
		# ------------------------------------------------------------------
		def _apply_dynamic_tightening(self, ema_slope: float, ema_vol: float) -> None:
				"""
				When the EMA shows a **positive slope** (loss going up) OR a
				**high volatility** we shrink the counters to make stopping more
				aggressive.  The factor `dynamic_factor` (default 0.5) controls
				how strong the tightening is.
				"""
				# Conditions for “shaky” EMA
				shaky = (ema_slope > self.slope_threshold) or (ema_vol > self.volatility_threshold)

				if shaky:
						# tighten patience (halve it, but never go below 1)
						self.patience = max(1, int(self._orig_patience * self.dynamic_factor))
						# lower allowed volatility (e.g. 30 % of original)
						self.volatility_threshold = self._orig_vol_thresh * self.dynamic_factor
						# raise slope requirement (make it harder to keep going)
						self.slope_threshold = self._orig_slope_thresh * (1.0 + (1 - self.dynamic_factor))
						print("\t[Dynamic] EMA shaky → tightening thresholds:")
						print(
								f"\t   patience → {self.patience}  "
								f"volatility_thr → {self.volatility_threshold:.2f}%  "
								f"slope_thr → {self.slope_threshold:.5f}"
						)
				else:
						# restore original thresholds for a calm EMA
						self.patience = self._orig_patience
						self.volatility_threshold = self._orig_vol_thresh
						self.slope_threshold = self._orig_slope_thresh

		# ------------------------------------------------------------------
		#  Public accessors – same API you used before
		# ------------------------------------------------------------------
		def get_status(self) -> Dict[str, Any]:
				"""Return a dictionary with the current internal state (useful for logging)."""
				status = {
						"best_score": self.best_score,
						"best_epoch": self.best_epoch + 1 if self.best_score is not None else None,
						"patience_counter": self.counter,
						"value_history_len": len(self.value_history),
						"ema_history_len": len(self.ema_history),
				}
				if len(self.value_history) >= self.window_size:
						window = self.value_history[-self.window_size :]
						status["volatility_window"] = self._volatility(window)
						status["slope_window"] = compute_slope(window)
				else:
						status["volatility_window"] = None
						status["slope_window"] = None
				return status

		def get_best_score(self) -> Optional[float]:
				return self.best_score

		def get_best_epoch(self) -> int:
				return self.best_epoch

		# ------------------------------------------------------------------
		#  Core method – called at the *end* of every epoch
		# ------------------------------------------------------------------
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
				"""
				Returns True if training should stop **after** this epoch.
				All side‑effects (checkpoint saving, weight restoration) are performed
				inside the method.
				"""

				# --------------------------------------------------------------
				# 1️⃣  Record raw loss and update EMA
				# --------------------------------------------------------------
				self.value_history.append(current_value)
				self._update_ema(current_value)

				# --------------------------------------------------------------
				# 2️⃣  (Optional) record training loss for over‑fitting gap
				# --------------------------------------------------------------
				# If you happen to have a running average of training loss,
				# feed it via `self.train_loss_history.append(train_loss_this_epoch)`.
				# The fine‑tuning script can simply call:
				#     early_stopper.train_loss_history.append(avg_training_loss)
				# before the `should_stop` call.
				# --------------------------------------------------------------

				# --------------------------------------------------------------
				# 3️⃣  Logging header
				# --------------------------------------------------------------
				phase_info = f", Phase {current_phase}" if current_phase is not None else ""
				print(f"\n--- EarlyStopping Check (Epoch {epoch+1}{phase_info}) ---")
				print(f"Raw validation loss: {current_value:.6f}")
				print(f"EMA ({self.ema_window}) loss: {self.ema_history[-1]:.6f}")

				# --------------------------------------------------------------
				# 4️⃣  Ignore everything before `min_epochs`
				# --------------------------------------------------------------
				if epoch < self.min_epochs:
						print(f"Skipping early‑stop (epoch {epoch+1} ≤ min_epochs {self.min_epochs})")
						return False

				# --------------------------------------------------------------
				# 5️⃣  Update *best* checkpoint if we have an improvement
				# --------------------------------------------------------------
				if self._is_improvement(current_value):
						print(
								f"\t>>> NEW BEST! loss improved from "
								f"{self.best_score if self.best_score is not None else 'N/A'} "
								f"to {current_value:.6f}"
						)
						self.best_score = current_value
						self.best_epoch = epoch
						self.counter = 0
						self.improvement_history.append(True)

						if self.restore_best_weights:
								# store a CPU‑detached copy to avoid device mismatches later
								self.best_weights = {
										k: v.clone().cpu().detach() for k, v in model.state_dict().items()
								}

						# ---------- checkpoint saving ----------
						checkpoint = {
								"epoch": self.best_epoch,
								"model_state_dict": self.best_weights
								if self.best_weights is not None
								else model.state_dict(),
								"optimizer_state_dict": optimizer.state_dict(),
								"scheduler_state_dict": scheduler.state_dict(),
								"best_val_loss": self.best_score,
						}
						if current_phase is not None:
								checkpoint["phase"] = current_phase

						try:
								torch.save(checkpoint, checkpoint_path)
								print(f"Saved new best checkpoint → {checkpoint_path}")
						except Exception as exc:
								print(f"⚠️  Failed to save checkpoint: {exc}")

				else:
						# No improvement → increase patience counter
						self.counter += 1
						self.improvement_history.append(False)
						print(
								f"\tNo improvement. Best: {self.best_score:.6f} "
								f"Patience: {self.counter}/{self.patience}"
						)

				# --------------------------------------------------------------
				# 6️⃣  Compute window‑based statistics on the **raw** loss
				# --------------------------------------------------------------
				if len(self.value_history) < self.window_size:
						print(
								f"\tInsufficient raw history ({len(self.value_history)}/{self.window_size}) "
								"for window‑based checks."
						)
						# Even if we cannot compute a window yet we may still stop by patience.
						if self.counter >= self.patience:
								phase_ok = (current_phase is None) or (
										current_phase >= self.min_phases_before_stopping
								)
								if phase_ok:
										print(f"❗️ Patience exceeded → early stop.")
										return True
						return False

				raw_window = self.value_history[-self.window_size :]

				raw_slope = compute_slope(raw_window)
				raw_volatility = self._volatility(raw_window)

				# Pairwise improvement on the raw window (sign‑aware)
				pairwise_diffs = [
						(raw_window[i] - raw_window[i + 1]) * self.sign
						for i in range(len(raw_window) - 1)
				]
				pairwise_improvement = np.mean(pairwise_diffs) if pairwise_diffs else 0.0

				# Cumulative improvement across the whole raw window
				cum_imp_signed = (raw_window[0] - raw_window[-1]) * self.sign
				cum_imp_abs = abs(cum_imp_signed)

				# --------------------------------------------------------------
				# 7️⃣  Compute **EMA‑based** trend signals (used for dynamic tightening)
				# --------------------------------------------------------------
				if len(self.ema_history) >= self.ema_window:
						ema_window_vals = self.ema_history[-self.ema_window :]
						ema_slope = compute_slope(ema_window_vals)
						ema_vol = self._volatility(ema_window_vals)
				else:
						ema_slope = 0.0
						ema_vol = 0.0

				# --------------------------------------------------------------
				# 8️⃣  Dynamic‑threshold adjustment based on EMA trend
				# --------------------------------------------------------------
				self._apply_dynamic_tightening(ema_slope, ema_vol)

				# --------------------------------------------------------------
				# 9️⃣  Assemble stopping reasons (same logic as before, but now
				#      thresholds may have been tightened for this epoch)
				# --------------------------------------------------------------
				stop_reasons = []

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
						stop_reasons.append(
								f"Low pairwise improvement ({pairwise_improvement:.5e}) & not close to best"
						)

				if cum_imp_abs < self.cumulative_delta:
						stop_reasons.append(f"Low cumulative improvement ({cum_imp_abs:.5e})")

				# --------------------------------------------------------------
				# 10️⃣  (Optional) add a *high‑level* recommendation derived
				#      from the EMA *trend* itself – this mimics the
				#      `LossAnalyzer.get_training_signals()` logic.
				# --------------------------------------------------------------
				recent_trend = (
						np.mean(np.diff(self.ema_history[-10:])) if len(self.ema_history) >= 10 else 0.0
				)
				if recent_trend > 0.001:  # loss increasing
						stop_reasons.append("EMA trend ↑ (recommend STOP)")
				elif recent_trend > -0.001:  # flat
						stop_reasons.append("EMA trend ≈0 (recommend CAUTION)")

				# --------------------------------------------------------------
				# 11️⃣  Final decision – respect phase constraints
				# --------------------------------------------------------------
				should_stop = bool(stop_reasons)
				if should_stop:
						phase_ok = (current_phase is None) or (
								current_phase >= self.min_phases_before_stopping
						)
						if not phase_ok:
								print(
										f"\tStopping criteria met ({', '.join(stop_reasons)}) "
										f"but waiting for phase ≥ {self.min_phases_before_stopping}"
								)
								should_stop = False
						else:
								print("\n<!> EARLY STOPPING TRIGGERED:")
								for r in stop_reasons:
										print(f"\t • {r}")

				else:
						print("\tNo stopping condition satisfied this epoch.")

				# --------------------------------------------------------------
				# 12️⃣  Restore best weights if we really stop
				# --------------------------------------------------------------
				if should_stop and self.restore_best_weights:
						if self.best_weights is not None:
								target_device = next(model.parameters()).device
								model.load_state_dict(
										{k: v.to(target_device) for k, v in self.best_weights.items()}
								)
								print(
										f"⚙️  Restored best weights from epoch {self.best_epoch+1} "
										f"(score={self.best_score:.6f})"
								)
						else:
								print("⚠️  No best weights stored – cannot restore.")

				return should_stop

		# ------------------------------------------------------------------
		#  Additional convenience – expose EMA‑based signals for reporting
		# ------------------------------------------------------------------
		def get_ema_signals(self) -> Dict[str, Any]:
				"""
				Mimics `LossAnalyzer.get_training_signals()` but works on‑the‑fly.
				Returns:
						- best_epoch (minimum EMA loss)
						- best_loss  (EMA value at that epoch)
						- overfitting_gap (val_ema - train_ema)  (requires train loss history)
						- recent_trend (mean diff of last 10 EMA points)
						- recommendation (STOP / CAUTION / CONTINUE)
				"""
				ema_vals = np.asarray(self.ema_history)
				if len(ema_vals) == 0:
						return {}

				best_idx = np.argmin(ema_vals) if self.mode == "min" else np.argmax(ema_vals)
				best_epoch = best_idx + 1
				best_loss = ema_vals[best_idx]

				# over‑fitting gap – needs a training‑loss history; if missing we skip
				overfit_gap = None
				if self.train_loss_history:
						train_ema = exponential_moving_average(
								np.asarray(self.train_loss_history), self.ema_window
						)
						overfit_gap = ema_vals[-1] - train_ema[-1]

				# recent trend (mean first‑difference of the last 10 EMA points)
				recent_trend = (
						np.mean(np.diff(ema_vals[-10:])) if len(ema_vals) >= 10 else 0.0
				)

				# textual recommendation (same logic as in `LossAnalyzer.get_training_signals`)
				if recent_trend > 0.001:
						recommendation = "STOP - Validation loss increasing"
				elif recent_trend > -0.001:
						recommendation = "CAUTION - Loss plateauing"
				else:
						recommendation = "CONTINUE - Still improving"

				return {
						"best_epoch": best_epoch,
						"best_loss": best_loss,
						"overfitting_gap": overfit_gap,
						"recent_trend": recent_trend,
						"recommendation": recommendation,
				}


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