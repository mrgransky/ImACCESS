from utils import *

def _phase_cmap(num_phases: int) -> np.ndarray:
	return plt.cm.Set3(np.linspace(0, 1, max(num_phases, 1)))

def _plot_loss_evolution(
		training_history: Dict,
		phase_colors: np.ndarray,
		save_path: str,
	) -> Tuple[plt.Figure, Dict]:
	epochs = [e + 1 for e in training_history["epochs"]]
	train_losses = training_history["train_losses"]
	val_losses = training_history["val_losses"]
	phases = training_history["phases"]
	transitions = training_history.get("phase_transitions", [])
	early_stop = training_history.get("early_stop_epoch")
	best_epoch = training_history.get("best_epoch")
	fig, ax = plt.subplots(figsize=(18, 14), facecolor="white")
	# ---- background shading per phase ---------------------------------
	for phase in set(phases):
			mask = [p == phase for p in phases]
			phase_epochs = np.array(epochs)[mask]
			if phase_epochs.size:
					ax.axvspan(
							phase_epochs.min(),
							phase_epochs.max(),
							color=phase_colors[phase],
							alpha=0.39,
							label=f"Phase {phase}",
					)
	# ---- loss curves ---------------------------------------------------
	ax.plot(
			epochs,
			train_losses,
			color="#0025FA",
			linewidth=2.5,
			marker="o",
			markersize=2,
			label="Training loss",
	)
	ax.plot(
			epochs,
			val_losses,
			color="#C77203",
			linewidth=2.5,
			marker="s",
			markersize=2,
			label="Validation loss",
	)
	# ---- transition markers ---------------------------------------------
	for i, tr in enumerate(transitions):
			ax.axvline(tr, color="#E91111", linestyle="--", linewidth=1.5, alpha=0.8, zorder=10)
			if tr < len(val_losses):
					before = val_losses[tr - 1] if tr > 0 else val_losses[tr]
					after = val_losses[tr]
					change = ((before - after) / before) * 100 if before > 0 else 0
					txt = f"Transition {i+1} ({change:+.2f}%)"
					ax.text(
							tr + 0.4,
							max(val_losses) * 1.01,
							txt,
							rotation=90,
							fontsize=9,
							color="#E91111",
							ha="left",
							va="bottom",
					)
	# ---- best model & early‑stop markers --------------------------------
	if best_epoch is not None and best_epoch < len(epochs):
			ax.scatter(
					epochs[best_epoch],
					val_losses[best_epoch],
					s=150,
					marker="*",
					color="#D8BA10",
					edgecolor="black",
					linewidth=1.5,
					zorder=15,
					label="Best model",
			)
	if early_stop is not None:
			ax.axvline(
					early_stop,
					color="#1D0808",
					linestyle=":",
					linewidth=2,
					alpha=0.9,
					label="Early stopping",
			)
			ax.text(
					early_stop + 0.4,
					max(val_losses) * 1.01,
					"Early stop",
					rotation=90,
					fontsize=9,
					color="#1D0808",
					ha="left",
					va="bottom",
			)
	ax.set_xlabel("Epoch", fontsize=10)
	ax.set_ylabel("Loss", fontsize=10)
	ax.set_title("Loss Evolution with Phase Transitions", fontsize=12, weight="bold")
	ax.grid(True, alpha=0.4)
	ax.legend(loc="best", fontsize=9, ncol=len(transitions) + 5, frameon=False)
	# ylim with a 25 % margin
	y_max = max(max(train_losses), max(val_losses))
	y_min = min(min(train_losses), min(val_losses))
	margin = 0.25 * y_max
	ax.set_ylim(y_min - margin, y_max + margin)
	fig.savefig(save_path, dpi=300, bbox_inches="tight")
	plt.close(fig)

	return fig, {
		"epochs": epochs,
		"train_losses": train_losses,
		"val_losses": val_losses,
		"best_epoch": best_epoch,
		"early_stop": early_stop,
	}

def _plot_lr_evolution(
		epochs: List[int],
		learning_rates: List[float],
		phases: List[int],
		phase_colors: np.ndarray,
		transitions: List[int],
		save_path: str,
) -> plt.Figure:
		fig, ax = plt.subplots(figsize=(12, 4), facecolor="white")

		for i in range(len(epochs) - 1):
				p = phases[i]
				ax.plot(
						[epochs[i], epochs[i + 1]],
						[learning_rates[i], learning_rates[i + 1]],
						color=phase_colors[p],
						linewidth=2.5,
						alpha=0.8,
				)

		for tr in transitions:
				ax.axvline(tr, color="#E91111", linestyle="--", linewidth=1.5, alpha=0.7)

		ax.set_xlabel("Epoch", fontsize=10, weight="bold")
		ax.set_ylabel("Learning Rate (log)", fontsize=10, weight="bold")
		ax.set_title("Learning‑Rate Adaptation Across Phases", fontsize=12, weight="bold")
		ax.set_yscale("log")
		ax.grid(True, alpha=0.3)

		fig.savefig(save_path, dpi=300, bbox_inches="tight")
		plt.close(fig)
		return fig

def _plot_weight_decay_evolution(
		epochs: List[int],
		weight_decays: List[float],
		phases: List[int],
		phase_colors: np.ndarray,
		transitions: List[int],
		save_path: str,
) -> plt.Figure:
		fig, ax = plt.subplots(figsize=(12, 4), facecolor="white")

		for i in range(len(epochs) - 1):
				p = phases[i]
				ax.plot(
						[epochs[i], epochs[i + 1]],
						[weight_decays[i], weight_decays[i + 1]],
						color=phase_colors[p],
						linewidth=2.5,
						alpha=0.8,
				)

		for tr in transitions:
				ax.axvline(tr, color="#E91111", linestyle="--", linewidth=1.5, alpha=0.7)

		ax.set_xlabel("Epoch", fontsize=10, weight="bold")
		ax.set_ylabel("Weight Decay (log)", fontsize=10, weight="bold")
		ax.set_title("Weight‑Decay Adaptation", fontsize=12, weight="bold")
		ax.set_yscale("log")
		ax.grid(True, alpha=0.3)

		fig.savefig(save_path, dpi=300, bbox_inches="tight")
		plt.close(fig)
		return fig

def _plot_phase_efficiency(
		phases: List[int],
		epochs: List[int],
		val_losses: List[float],
		phase_colors: np.ndarray,
		save_path: str,
	) -> plt.Figure:

	# ----- compute per‑phase statistics ---------------------------------
	uniq = sorted(set(phases))
	durations = []
	improvements = []
	for ph in uniq:
		idx = [i for i, p in enumerate(phases) if p == ph]
		dur = len(idx)
		durations.append(dur)
		# improvement within the phase (first → last val loss)
		start, end = idx[0], idx[-1]
		start_l, end_l = val_losses[start], val_losses[end]
		imp = ((start_l - end_l) / start_l * 100) if start_l > 0 else 0
		improvements.append(imp)
	
	fig, ax1 = plt.subplots(figsize=(10, 7), facecolor="white")
	bars = ax1.bar(
		np.arange(len(uniq)),
		durations,
		color=[phase_colors[p] for p in uniq],
		alpha=0.7,
		edgecolor="black",
	)
	ax1.set_xlabel("Phase", fontsize=10, weight="bold")
	ax1.set_ylabel("Duration (epochs)", color="tab:blue", fontsize=10, weight="bold")
	ax1.set_title("Phase Efficiency Analysis", fontsize=12, weight="bold")
	ax1.set_xticks(np.arange(len(uniq)))
	ax1.set_xticklabels([f"P{p}" for p in uniq])
	ax1.tick_params(axis="y", labelcolor="tab:blue")

	# ----- overlay improvement line --------------------------------------
	ax2 = ax1.twinx()
	ax2.plot(
		np.arange(len(uniq)),
		improvements,
		"ro-",
		linewidth=1.2,
		markersize=3,
		label="Loss improvement %"
	)
	ax2.set_ylabel("Loss Improvement (%)", color="tab:red", fontsize=10, weight="bold")
	ax2.tick_params(axis="y", labelcolor="tab:red")

	for i, (b, dur, imp) in enumerate(zip(bars, durations, improvements)):
		ax1.text(
			b.get_x() + b.get_width() / 2,
			b.get_height() + 0.2,
			f"{dur}",
			ha="center",
			va="bottom",
			fontsize=9,
			fontweight="bold",
		)
		ax2.text(
			i,
			imp + 0.8,
			f"{imp:.1f}%",
			ha="center",
			va="bottom",
			fontsize=9,
			fontweight="bold",
			color="darkred",
		)
	fig.tight_layout()
	fig.savefig(save_path, dpi=300, bbox_inches="tight")
	plt.close(fig)

	return fig

def _plot_hyperparameter_correlation(
		epochs: List[int],
		learning_rates: List[float],
		weight_decays: List[float],
		val_losses: List[float],
		transitions: List[int],
		save_path: str,
	) -> plt.Figure:
		# normalise to [0,1]
		lr_n = np.array(learning_rates) / max(learning_rates)
		wd_n = np.array(weight_decays) / max(weight_decays)
		loss_n = np.array(val_losses) / max(val_losses)

		fig, ax = plt.subplots(figsize=(12, 4), facecolor="white")
		ax.plot(epochs, lr_n, "g-", linewidth=2, label="LR (norm)")
		ax.plot(epochs, wd_n, "m-", linewidth=2, label="WD (norm)")
		ax.plot(epochs, loss_n, "r-", linewidth=2, label="Val loss (norm)")

		for tr in transitions:
			ax.axvline(tr, color="#E91111", linestyle="--", linewidth=1.5, alpha=0.7)

		ax.set_xlabel("Epoch", fontsize=10, weight="bold")
		ax.set_ylabel("Normalised value", fontsize=10, weight="bold")
		ax.set_title("Hyper‑parameter Correlations", fontsize=12, weight="bold")
		ax.set_ylim(0, 1.1)
		ax.grid(True, alpha=0.3)
		ax.legend(fontsize=9, loc="best", ncol=3, frameon=True, fancybox=True, shadow=True, edgecolor="black", facecolor="white")

		fig.savefig(save_path, dpi=300, bbox_inches="tight")
		plt.close(fig)
		return fig

def _plot_trainable_layers_progression(
		epochs: List[int],
		phases: List[int],
		unfreeze_schedule: Dict[int, List[str]],
		layer_groups: Dict[str, List[str]],
		phase_colors: np.ndarray,
		transitions: List[int],
		save_path: str,
) -> plt.Figure:
		# ------------------------------------------------------------------
		# 1️⃣  Determine the *true* number of phases
		# ------------------------------------------------------------------
		# Phases that actually occurred during training (from `phases`)
		# and phases that are mentioned in the unfreeze schedule (might be
		# ahead of the current training run, e.g. a future phase that will be
		# unfrozen later).  We need the union of both sets.
		phase_set = set(phases) | set(unfreeze_schedule.keys())
		if not phase_set:                     # defensive – should never happen
				raise ValueError("No phase information supplied.")
		max_phase = max(phase_set)            # highest phase index that exists
		n_phases = max_phase + 1               # length of the zero‑based array

		# ------------------------------------------------------------------
		# 2️⃣  Build the “how many layers are unfrozen” vector
		# ------------------------------------------------------------------
		unfrozen_per_phase = np.zeros(n_phases, dtype=int)
		for ph, layers in unfreeze_schedule.items():
				# if a phase appears in the schedule but not in the training run,
				# we still store the information – the plot will simply show a
				# horizontal line at the correct y‑value for that phase.
				unfrozen_per_phase[ph] = len(layers)

		# ------------------------------------------------------------------
		# 3️⃣  Prepare a mapping phase → list of epoch numbers belonging to it
		# ------------------------------------------------------------------
		phase_to_epochs: Dict[int, List[int]] = {}
		for ep, ph in zip(epochs, phases):
				phase_to_epochs.setdefault(ph, []).append(ep)

		# ------------------------------------------------------------------
		# 4️⃣  Plot the step‑wise progression
		# ------------------------------------------------------------------
		fig, ax = plt.subplots(figsize=(12, 4), facecolor="white")

		for ph, ep_list in phase_to_epochs.items():
				# colour comes from the common palette (`phase_colors`); guard against
				# an out‑of‑range index (possible when the palette was built only from
				# `max(phases) + 1` before the fix).  If the colour is missing, fall back
				# to a neutral gray.
				colour = phase_colors[ph] if ph < len(phase_colors) else "#bbbbbb"
				ax.hlines(
						unfrozen_per_phase[ph],
						min(ep_list),
						max(ep_list),
						colors=colour,
						linewidth=4,
						label=f"Phase {ph}",
				)

		# ------------------------------------------------------------------
		# 5️⃣  Add transition markers, axes, legend, etc.
		# ------------------------------------------------------------------
		for tr in transitions:
				ax.axvline(tr, color="#E91111", linestyle="--", linewidth=1.5, alpha=0.7)

		ax.set_xlabel("Epoch", fontsize=10, weight="bold")
		ax.set_ylabel("Trainable layers", fontsize=10, weight="bold")
		ax.set_title("Layer Un‑freezing Progression", fontsize=12, weight="bold")

		# Upper‑limit: total number of *individual* layers (not just groups)
		total_groups = sum(len(g) for g in layer_groups.values())
		ax.set_ylim(0, total_groups * 1.1)

		ax.grid(True, alpha=0.3)
		ax.legend(loc="best", fontsize=9, frameon=False)

		fig.savefig(save_path, dpi=300, bbox_inches="tight")
		plt.close(fig)
		return fig

def _plot_unfreeze_heatmap(
		unfreeze_schedule: dict,
		layer_groups: dict,
		max_phase: int,
		save_path: str,
	) -> plt.Figure:
	group_names = list(layer_groups.keys())
	n_groups = len(group_names)
	# heat-map matrix
	heat = np.zeros((n_groups, max_phase + 1))
	for ph in range(max_phase + 1):
			if ph not in unfreeze_schedule:
					continue
			unfrozen_set = set(unfreeze_schedule[ph])
			for g_idx, (g_name, g_layers) in enumerate(layer_groups.items()):
					n_total = len(g_layers)
					n_unfrozen = sum(1 for l in g_layers if any(u in l for u in unfrozen_set))
					heat[g_idx, ph] = n_unfrozen / max(n_total, 1)
	# Color Universal Design friendly colormap (light → mid → dark blue)
	cmap = LinearSegmentedColormap.from_list(
			"cud_safe_blue",
			[
					(0.00, "#f0f0f0"),  # light gray
					(0.25, "#c6dbef"),  # pale blue
					(0.50, "#6baed6"),  # medium blue
					(0.75, "#2171b5"),  # rich blue
					(1.00, "#08306b"),  # deep navy
			],
			N=256
	)
	fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
	im = ax.imshow(heat, cmap=cmap, aspect="auto", vmin=0, vmax=1)
	
	# Add vertical separators between phases
	for p in range(1, max_phase + 1):
		ax.axvline(p - 0.5, color="#FCB4B4", linewidth=1.2)
	
	# Axes & labels
	ax.set_xticks(np.arange(max_phase + 1))
	ax.set_xticklabels([f"P{p}" for p in range(max_phase + 1)], fontsize=9)
	ax.set_yticks(np.arange(n_groups))
	ax.set_yticklabels([name.replace("_", "\n").title() for name in group_names], fontsize=9)
	ax.set_xlabel("Phase", fontsize=10, weight="bold")
	ax.set_ylabel("Layer groups", fontsize=10, weight="bold")
	ax.set_title("Layer-Group Un-freezing Pattern", fontsize=14, weight="bold", pad=15)
	# Colorbar
	cbar = fig.colorbar(im, ax=ax, shrink=0.8)
	cbar.set_label("Fraction Unfrozen", fontsize=10)
	cbar.ax.tick_params(labelsize=8)
	# Save
	fig.savefig(save_path, dpi=300, bbox_inches="tight")
	plt.close(fig)
	return fig

def _write_training_summary(
		training_history: Dict,
		phase_colors: np.ndarray,
		save_path: str,
	) -> None:
	epochs = [e + 1 for e in training_history["epochs"]]
	train_losses = training_history["train_losses"]
	val_losses = training_history["val_losses"]
	phases = training_history["phases"]
	transitions = training_history.get("phase_transitions", [])
	early_stop = training_history.get("early_stop_epoch")
	best_epoch = training_history.get("best_epoch")
	total_epochs = len(epochs)
	num_phases = len(set(phases))
	best_val = min(val_losses) if val_losses else np.nan
	total_improvement = (
		(val_losses[0] - best_val) / val_losses[0] * 100
		if val_losses and val_losses[0] > 0
		else 0
	)
	avg_phase_len = np.mean([len([p for p in phases if p == ph]) for ph in set(phases)])
	# pick the most effective phase (largest % improvement)
	# – same logic as in the original function
	phase_stats = {}
	for ph in set(phases):
		idx = [i for i, p in enumerate(phases) if p == ph]
		if not idx:
			continue
		imp = (
			(val_losses[idx[0]] - val_losses[idx[-1]]) / val_losses[idx[0]] * 100
			if val_losses[idx[0]] > 0
			else 0
		)
		phase_stats[ph] = imp
	best_phase = max(phase_stats, key=phase_stats.get) if phase_stats else -1
	lines = [
		"TRAINING SUMMARY",
		f"  • Total epochs          : {total_epochs}",
		f"  • Number of phases      : {num_phases}",
		f"  • Phase transitions     : {len(transitions)}",
		f"  • Avg. phase duration   : {avg_phase_len:.1f} epochs",
		f"  • Best phase (most loss improvement) : Phase {best_phase}",
		f"  • Total loss improvement: {total_improvement:.2f} %",
		f"  • Final training loss   : {train_losses[-1]:.6f}",
		f"  • Final validation loss : {val_losses[-1]:.6f}",
		f"  • Best validation loss  : {best_val:.6f}",
	]
	if early_stop:
		lines.append(f"  • Early stopped at epoch: {early_stop}")
	if best_epoch is not None:
		lines.append(f"  • Best model epoch      : {epochs[best_epoch]} (val loss {val_losses[best_epoch]:.4f})")
	txt = "\n".join(lines) + "\n"
	with open(save_path, "w", encoding="utf‑8") as fp:
		fp.write(txt)

def plot_progressive_fine_tuning_report(
		training_history: Dict,
		unfreeze_schedule: Dict[int, List[str]],
		layer_groups: Dict[str, List[str]],
		plot_paths: Dict[str, str],
	) -> Dict[str, Optional[plt.Figure]]:
	epochs = [e + 1 for e in training_history["epochs"]]
	phases = training_history["phases"]
	transitions = training_history.get("phase_transitions", [])
	phase_colors = _phase_cmap(max(phases) + 1)
	fig_loss, loss_meta = _plot_loss_evolution(
		training_history=training_history,
		phase_colors=phase_colors,
		save_path=plot_paths["loss_evolution"],
	)

	fig_lr = _plot_lr_evolution(
		epochs=epochs,
		learning_rates=training_history["learning_rates"],
		phases=phases,
		phase_colors=phase_colors,
		transitions=transitions,
		save_path=plot_paths["lr_evolution"],
	)

	fig_wd = _plot_weight_decay_evolution(
		epochs=epochs,
		weight_decays=training_history["weight_decays"],
		phases=phases,
		phase_colors=phase_colors,
		transitions=transitions,
		save_path=plot_paths["wd_evolution"],
	)

	fig_phase_eff = _plot_phase_efficiency(
		phases=phases,
		epochs=epochs,
		val_losses=training_history["val_losses"],
		phase_colors=phase_colors,
		save_path=plot_paths["phase_efficiency"],
	)

	fig_corr = _plot_hyperparameter_correlation(
		epochs=epochs,
		learning_rates=training_history["learning_rates"],
		weight_decays=training_history["weight_decays"],
		val_losses=training_history["val_losses"],
		transitions=transitions,
		save_path=plot_paths["hyperparameter_correlation"],
	)

	fig_trainable = _plot_trainable_layers_progression(
		epochs=epochs,
		phases=phases,
		unfreeze_schedule=unfreeze_schedule,
		layer_groups=layer_groups,
		phase_colors=phase_colors,
		transitions=transitions,
		save_path=plot_paths["trainable_layers"],
	)

	fig_heat = _plot_unfreeze_heatmap(
		unfreeze_schedule=unfreeze_schedule,
		layer_groups=layer_groups,
		max_phase=max(phases),
		save_path=plot_paths["unfreeze_heatmap"],
	)

	_write_training_summary(
		training_history=training_history,
		phase_colors=phase_colors,
		save_path=plot_paths["training_summary"],
	)

	return {
		"loss": fig_loss,
		"learning_rate": fig_lr,
		"weight_decay": fig_wd,
		"phase_efficiency": fig_phase_eff,
		"hyperparameter_correlation": fig_corr,
		"trainable_layers": fig_trainable,
		"unfreeze_heatmap": fig_heat,
	}

def plot_phase_transition_analysis(
		training_history: Dict,
		save_path: str,
		figsize: Tuple[int, int] = (20, 10)
	):
	# Extract data
	epochs = training_history['epochs']
	epochs = list(map(lambda x: x + 1, epochs)) # convert to 1-based indexing
	train_losses = training_history['train_losses']
	val_losses = training_history['val_losses']
	learning_rates = training_history['learning_rates']
	weight_decays = training_history['weight_decays']
	phases = training_history['phases']
	transitions = training_history.get('phase_transitions', [])
	early_stop_epoch = training_history.get('early_stop_epoch')
	best_epoch = training_history.get('best_epoch')
	
	# Create figure with custom layout
	fig = plt.figure(figsize=figsize, facecolor='white')
	gs = fig.add_gridspec(
		3, 
		3, 
		height_ratios=[2, 1.5, 1.2], 
		width_ratios=[2, 1, 1.2], 
		hspace=0.35, 
		wspace=0.4,
	)
	
	# Color scheme
	phase_colors = plt.cm.Set3(np.linspace(0, 1, max(phases) + 1))
	transition_color = "#E91111"
	early_stop_color = "#1D0808"
	best_model_color = "#D8BA10"
	
	# ================================
	# 1. Main Loss Evolution Plot (top-left, spans 2 columns)
	# ================================
	ax1 = fig.add_subplot(gs[0, :])
	
	# Add phase background shading
	for phase in set(phases):
		phase_epochs = [e for e, p in zip(epochs, phases) if p == phase]
		if phase_epochs:
			start_epoch = min(phase_epochs)
			end_epoch = max(phase_epochs)
			ax1.axvspan(
				start_epoch, 
				end_epoch, 
				alpha=0.39,
				color=phase_colors[phase], 
				label=f'Phase {phase}'
			)
	
	# Plot loss curves with enhanced styling
	train_line = ax1.plot(
		epochs, 
		train_losses, 
		color="#0025FA",
		linestyle='-',
		linewidth=2.5, 
		label='Training Loss', 
		alpha=0.9, 
		marker='o', 
		markersize=1.8,
	)
	val_line = ax1.plot(
		epochs, 
		val_losses, 
		color="#C77203",
		linestyle='-',
		linewidth=2.5, 
		label='Validation Loss', 
		alpha=0.9, 
		marker='s', 
		markersize=1.8,
	)
	
	# Mark phase transitions with enhanced annotations
	for i, transition_epoch in enumerate(transitions):
		ax1.axvline(
			x=transition_epoch, 
			color=transition_color, 
			linestyle='--', 
			linewidth=1.5,
			alpha=0.8,
			zorder=10,
		)
		
		if transition_epoch < len(val_losses):
			transition_loss = val_losses[transition_epoch]
			improvement_text = ""
			if transition_epoch > 0:
				prev_loss = val_losses[transition_epoch - 1]
				change = ((prev_loss - transition_loss) / prev_loss) * 100
				improvement_text = f"\n{change:+.2f} %"
			
			ax1.text(
				transition_epoch + 0.6,
				max(val_losses) * 1.02,
				f'Transition {i+1}{improvement_text}',
				rotation=90,
				fontsize=10,
				ha='left',
				va='bottom',
				color=transition_color,
				bbox=dict(
					boxstyle="round,pad=0.4",
					edgecolor='none',
					facecolor='none',
					alpha=0.9,
				)
			)

	if best_epoch is not None and best_epoch < len(epochs):
		best_loss = val_losses[best_epoch]
		ax1.scatter(
			[epochs[best_epoch]], 
			[best_loss], 
			color=best_model_color, 
			s=150, 
			marker='*', 
			zorder=15, 
			label='Best Model', 
			edgecolor='black', 
			linewidth=1.5,
		)
	
	if early_stop_epoch is not None:
		ax1.axvline(
			x=early_stop_epoch, 
			color=early_stop_color, 
			linestyle=':',
			linewidth=1.8,
			alpha=0.9,
			label='Early Stopping',
			zorder=12
		)
		ax1.text(
			early_stop_epoch + 0.5,
			max(val_losses) * 1.01,
			'Early Stopping', 
			rotation=90, 
			va='bottom',
			ha='left',
			fontsize=10,
			color=early_stop_color, 
		)
	
	ax1.set_xlabel('Epoch', fontsize=8)
	ax1.set_ylabel('Loss', fontsize=8)
	ax1.set_title(f'Learning Curve with Phase Transitions', fontsize=8, weight='bold')
	ax1.legend(
		loc='best', 
		fontsize=8, 
		edgecolor='none',
		ncol=len(transitions)+5,
	)
	ax1.grid(True, alpha=0.5)

	# Set y-axis limits with minimum of 0 and maximum with margin
	max_loss = max(max(train_losses), max(val_losses))
	min_loss = min(min(train_losses), min(val_losses))
	margin = max_loss * 0.25  # 25% margin
	ax1.set_ylim(min_loss - margin, max_loss + margin)
	
	# ================================
	# 2. Learning Rate Adaptation
	# ================================
	ax2 = fig.add_subplot(gs[1, :1])
	
	# Plot learning rate with phase coloring
	for i in range(len(epochs)-1):
		phase = phases[i]
		ax2.semilogy(
			[epochs[i], epochs[i+1]], 
			[learning_rates[i], learning_rates[i+1]], 
			color=phase_colors[phase], 
			linewidth=3, 
			alpha=0.95,
		)
	
	# Mark transitions
	for transition_epoch in transitions:
		if transition_epoch < len(learning_rates):
			ax2.axvline(
				x=transition_epoch, 
				color=transition_color, 
				linewidth=2.5,
				alpha=0.55,
				linestyle='--',
			)
	
	ax2.set_xlabel('Epoch', fontsize=8, weight='bold')
	ax2.set_ylabel('Learning Rate (log)', fontsize=8, weight='bold')
	ax2.set_title('Learning Rate Adaptation Across Phases', fontsize=8, weight='bold')
	ax2.grid(True, alpha=0.3)
	
	# ================================
	# 3. Weight Decay Adaptation
	# ================================
	ax3 = fig.add_subplot(gs[1, 1:])
	
	# Plot weight decay with phase coloring
	for i in range(len(epochs)-1):
			phase = phases[i]
			ax3.semilogy(
				[epochs[i], epochs[i+1]], 
				[weight_decays[i], weight_decays[i+1]], 
				color=phase_colors[phase], 
				linewidth=3, 
				alpha=0.8
			)
	
	# Mark transitions
	for transition_epoch in transitions:
		if transition_epoch < len(weight_decays):
			ax3.axvline(
				x=transition_epoch, 
				color=transition_color, 
				linestyle='--', 
				linewidth=2, 
				alpha=0.7
			)
	
	ax3.set_xlabel('Epoch', fontsize=8, weight='bold')
	ax3.set_ylabel('Weight Decay (log)', fontsize=8, weight='bold')
	ax3.set_title('Weight Decay Adaptation Across Phases', fontsize=8, weight='bold')
	ax3.grid(True, alpha=0.3)
	
	# ================================
	# 4. Phase Duration and Efficiency Analysis (middle-center)
	# ================================
	ax4 = fig.add_subplot(gs[2, :1])
	
	# Calculate phase durations and improvements
	phase_data = []
	unique_phases = sorted(set(phases))
	
	for phase in unique_phases:
		phase_epochs = [e for e, p in zip(epochs, phases) if p == phase]
		duration = len(phase_epochs)
		
		# Calculate loss improvement in this phase
		if phase_epochs:
			start_idx = phase_epochs[0]
			end_idx = phase_epochs[-1]
			if start_idx < len(val_losses) and end_idx < len(val_losses):
				start_loss = val_losses[start_idx]
				end_loss = val_losses[end_idx]
				improvement = ((start_loss - end_loss) / start_loss * 100) if start_loss > 0 else 0
			else:
				improvement = 0
		else:
			improvement = 0
		
		phase_data.append((phase, duration, improvement))
	
	phases_list, durations, improvements = zip(*phase_data) if phase_data else ([], [], [])
	
	# Create dual-axis plot
	bars = ax4.bar(
		range(len(durations)), 
		durations,
		color=[phase_colors[p] for p in phases_list], 
		alpha=0.8,
	)
	
	# Add improvement percentages
	ax4_twin = ax4.twinx()
	improvement_line = ax4_twin.plot(
		range(len(improvements)), 
		improvements,
		'ro-',
		linewidth=1.0,
		markersize=2,
		label='Loss Improvement %'
	)
	
	for i, (bar, duration, improvement) in enumerate(zip(bars, durations, improvements)):
		# Duration labels on bars
		ax4.text(
			bar.get_x() + bar.get_width()/2., 
			bar.get_height() + 6,
			f'{duration}', 
			ha='center',
			va='bottom',
			fontweight='bold',
			fontsize=8,
			color='#0004EC',
		)
		# Improvement labels
		ax4_twin.text(
			i, 
			improvement + 0.3,
			f'{improvement:.2f}%',
			ha='center',
			va='bottom',
			fontweight='bold',
			fontsize=8,
			color="#F73100",
		)
	
	ax4.set_xlabel('Phase', fontsize=8, weight='bold')
	ax4.set_ylabel('Duration (Epochs)', fontsize=8, weight='bold', color='#0004EC')
	ax4_twin.set_ylabel('Loss Improvement (%)', fontsize=8, weight='bold', color="#F73100")
	ax4.set_title('Phase Efficiency Analysis', fontsize=8, weight='bold')
	
	phase_labels = [f'P{p}' for p in phases_list]
	ax4.set_xticks(range(len(phase_labels)))
	ax4.set_xticklabels(phase_labels)
	ax4.tick_params(axis='y', labelcolor='#0004EC')
	ax4_twin.tick_params(axis='y', labelcolor="#F73100")
	ax4_twin.spines['top'].set_visible(False)
	ax4.spines['top'].set_visible(False)
	
	# ================================
	# 5. Hyperparameter Correlation Analysis (middle-right)
	# ================================
	ax5 = fig.add_subplot(gs[2, 1:])
	
	# Normalize data for correlation plot
	lr_norm = np.array(learning_rates) / max(learning_rates)
	wd_norm = np.array(weight_decays) / max(weight_decays)
	loss_norm = np.array(val_losses) / max(val_losses)
	
	ax5.plot(epochs, lr_norm, 'g-', linewidth=1.2, label='LR', alpha=0.8)
	ax5.plot(epochs, wd_norm, 'm-', linewidth=1.2, label='WD', alpha=0.8)
	ax5.plot(epochs, loss_norm, 'r-', linewidth=1.2, label='Val Loss', alpha=0.8)
	
	# Mark transitions
	for transition_epoch in transitions:
		ax5.axvline(
			x=transition_epoch, 
			color=transition_color, 
			linestyle='--', 
			linewidth=1.5, 
			alpha=0.7,
		)
	
	ax5.set_xlabel('Epoch', fontsize=8, weight='bold')
	ax5.set_ylabel('Normalized Values', fontsize=8, weight='bold')
	ax5.set_title('Hyperparameter Correlations [normed]', fontsize=8, weight='bold')
	ax5.legend(
		fontsize=8,
		loc='best',
		ncol=3,
		frameon=True,
		fancybox=True,
		shadow=True,
		edgecolor='none',
		facecolor='white',
	)
	ax5.grid(True, alpha=0.3)
	ax5.set_ylim(0, 1.1)
	
	# ================================
	# 6. Training Statistics and Insights
	# ================================
	# Calculate comprehensive statistics
	total_epochs = len(epochs)
	num_phases = len(set(phases))
	total_improvement = ((val_losses[0] - min(val_losses)) / val_losses[0] * 100) if val_losses and val_losses[0] > 0 else 0
	avg_phase_duration = np.mean(durations) if durations else 0
	best_phase = phases_list[np.argmax(improvements)] if improvements else 0
	
	# Phase transition effectiveness
	transition_improvements = []
	for i, t_epoch in enumerate(transitions):
		if t_epoch > 0 and t_epoch < len(val_losses) - 1:
			before = val_losses[t_epoch - 1]
			after = val_losses[t_epoch + 1] if t_epoch + 1 < len(val_losses) else val_losses[t_epoch]
			improvement = ((before - after) / before * 100) if before > 0 else 0
			transition_improvements.append(improvement)
	
	avg_transition_improvement = np.mean(transition_improvements) if transition_improvements else 0
	
	# Learning rate adaptation analysis
	lr_changes = []
	for t_epoch in transitions:
		if t_epoch > 0 and t_epoch < len(learning_rates):
			before_lr = learning_rates[t_epoch - 1]
			after_lr = learning_rates[t_epoch]
			change = ((after_lr - before_lr) / before_lr * 100) if before_lr > 0 else 0
			lr_changes.append(change)
	
	# Create comprehensive summary
	summary_text = f"""
	COMPREHENSIVE TRAINING ANALYSIS:
	
	OVERALL PERFORMANCE:
	• Total Epochs: {total_epochs}
	• Number of Phases: {num_phases}
	• Total Loss Improvement: {total_improvement:.2f}%
	• Training Status: {'Early Stopped' if early_stop_epoch else 'Completed'}
	
	PHASE TRANSITION ANALYSIS:
	• Total Transitions: {len(transitions)}
	• Average Phase Duration: {avg_phase_duration:.1f} epochs
	• Most Effective Phase: Phase {best_phase}
	• Avg Improvement per Transition: {avg_transition_improvement:.2f}%
	
	HYPERPARAMETER ADAPTATION:
	• Initial Learning Rate: {learning_rates[0]:.2e}
	• Final Learning Rate: {learning_rates[-1]:.2e}
	• LR Reduction Factor: {(learning_rates[0]/learning_rates[-1]):.1f}x
	• Weight Decay Range: {min(weight_decays):.2e} → {max(weight_decays):.2e}
	"""
	
	if transitions:
		summary_text += f"\n    TRANSITION EPOCHS: {transitions}"
	
	if best_epoch is not None:
		summary_text += f"\n    Best Model: Epoch {epochs[best_epoch]} (Loss: {val_losses[best_epoch]:.4f})"
	
	# Phase-specific insights
	phase_insights = "\n    PHASE INSIGHTS:\n"
	for phase, duration, improvement in phase_data[:3]:  # Show top 3 phases
		phase_insights += f"    • Phase {phase}: {duration} epochs, {improvement:.1f}% improvement\n"
	
	summary_text += phase_insights
	
	print(f"Phase transition analysis summary:\n{summary_text}\n")
			
	plt.suptitle(
		f'Progressive Layer Unfreezing\nPhase Transition Analysis', 
		fontsize=11,
		weight='bold',
	)
	
	plt.savefig(
		save_path, 
		dpi=300, 
		bbox_inches='tight', 
		facecolor='white', 
		edgecolor='none',
	)
	
	plt.close()
	
	# Return analysis results for further use
	analysis_results = {
		'total_improvement': total_improvement,
		'num_transitions': len(transitions),
		'avg_phase_duration': avg_phase_duration,
		'best_phase': best_phase,
		'transition_improvements': transition_improvements,
		'lr_adaptation_factor': learning_rates[0]/learning_rates[-1] if learning_rates[-1] > 0 else 1.0
	}
	
	return analysis_results

def plot_progressive_training_dynamics(
		training_history: Dict,
		unfreeze_schedule: Dict[int, List[str]],
		layer_groups: Dict[str, List[str]],
		save_path: str,
		figsize: Tuple[int, int] = (18, 16)
	):
	"""
	Plot comprehensive progressive training dynamics including:
	- Training and validation losses
	- Learning rate and weight decay evolution
	- Layer unfreezing progression
	- Phase transitions
	- Early stopping events
	
	Args:
		training_history: Dictionary containing training metrics per epoch:
			{
				'epochs': List[int],
				'train_losses': List[float],
				'val_losses': List[float],
				'learning_rates': List[float],
				'weight_decays': List[float],
				'phases': List[int],
				'phase_transitions': List[int],  # Epochs where phase transitions occurred
				'early_stop_epoch': Optional[int],
				'best_epoch': Optional[int]
			}
		unfreeze_schedule: Phase -> layers mapping from progressive training
		layer_groups: Layer group definitions from get_layer_groups()
		save_path: Path to save the plot
		figsize: Figure size tuple
	"""
	# print(json.dumps(training_history, indent=4, ensure_ascii=False))

	fig = plt.figure(figsize=figsize, facecolor='white')
	gs = fig.add_gridspec(
		4, 
		2, 
		height_ratios=[2, 1.5, 1.5, 2], 
		width_ratios=[3, 1],
		hspace=0.3, 
		wspace=0.3,
	)
	
	epochs = training_history['epochs']
	epochs = list(map(lambda x: x + 1, epochs)) # convert to 1-based indexing
	train_losses = training_history['train_losses']
	val_losses = training_history['val_losses']
	learning_rates = training_history['learning_rates']
	weight_decays = training_history['weight_decays']
	phases = training_history['phases']
	phase_transitions = training_history.get('phase_transitions', [])
	early_stop_epoch = training_history.get('early_stop_epoch')
	best_epoch = training_history.get('best_epoch')
	
	phase_colors = plt.cm.Set3(np.linspace(0, 1, max(phases) + 1))
	
	# ==================
	# 1. Main Loss Plot
	# ==================
	ax1 = fig.add_subplot(gs[0, :])
	
	for i, phase in enumerate(set(phases)):
		phase_epochs = [e for e, p in zip(epochs, phases) if p == phase]
		if phase_epochs:
			start_epoch = min(phase_epochs)
			end_epoch = max(phase_epochs)
			ax1.axvspan(
				start_epoch,
				end_epoch,
				alpha=0.25,
				color=phase_colors[phase],
				label=f'Phase {phase}'
			)
	
	line1 = ax1.plot(
		epochs, 
		train_losses, 
		'b-', 
		linewidth=2.0, 
		label='Training Loss', 
		alpha=0.8,
	)
	line2 = ax1.plot(
		epochs,
		val_losses,
		'r-',
		linewidth=2.0,
		label='Validation Loss', 
		alpha=0.8,
	)
	
	# Mark phase transitions
	for transition_epoch in phase_transitions:
		ax1.axvline(
			x=transition_epoch, 
			color='orange', 
			linestyle='--',
			linewidth=1.5,
			alpha=0.7,
		)
		ax1.annotate(
			f'Phase\nTransition', xy=(transition_epoch, max(val_losses)*0.9),
			xytext=(transition_epoch+2, max(val_losses)*0.95),
			arrowprops=dict(arrowstyle='->', color='orange', alpha=0.7),
			fontsize=9, 
			ha='left', 
			color='orange', 
			weight='bold'
		)
	
	# Mark best epoch and early stopping
	if best_epoch is not None:
		best_loss = val_losses[best_epoch]
		ax1.scatter(
			[epochs[best_epoch]], 
			[best_loss], 
			color='gold', 
			s=100, 
			marker='*', 
			zorder=5, 
			label='Best Model', 
			edgecolor='black',
		)
	
	if early_stop_epoch is not None:
		ax1.axvline(
			x=early_stop_epoch, 
			color='red', 
			linestyle=':', 
			linewidth=2.5,
			alpha=0.8, 
			label='Early Stop'
		)
	
	ax1.set_xlabel('Epoch', fontsize=10, weight='bold')
	ax1.set_ylabel('Loss', fontsize=10, weight='bold')
	ax1.set_title(f'Learning Curve with Phase Transitions', fontsize=10, weight='bold')
	ax1.legend(
		loc='best', 
		fontsize=10,
		ncol=5+len(phase_transitions), 
		edgecolor='none',
	)
	ax1.grid(True, alpha=0.4)
	
	# ================================
	# 2. Learning Rate Evolution (middle-left)
	# ================================
	ax2 = fig.add_subplot(gs[1, 0])
	
	for i in range(len(epochs)-1):
		phase = phases[i]
		ax2.plot(
			[epochs[i], epochs[i+1]],
			[learning_rates[i], learning_rates[i+1]],
			color=phase_colors[phase],
			linewidth=2.5,
			alpha=0.8,
		)
	
	# Mark phase transitions
	for transition_epoch in phase_transitions:
		ax2.axvline(
			x=transition_epoch, 
			color='orange',
			linestyle='--',
			linewidth=1.5,
			alpha=0.7,
		)
	
	ax2.set_xlabel('Epoch', fontsize=11, weight='bold')
	ax2.set_ylabel('Learning Rate', fontsize=11, weight='bold')
	ax2.set_title('Learning Rate Adaptation', fontsize=12, weight='bold')
	ax2.set_yscale('log')
	ax2.grid(True, alpha=0.3)
	
	# ================================
	# 3. Weight Decay Evolution (middle-right)
	# ================================
	ax3 = fig.add_subplot(gs[1, 1])
	
	# Color code by phase
	for i in range(len(epochs)-1):
		phase = phases[i]
		ax3.plot(
			[epochs[i], epochs[i+1]], 
			[weight_decays[i], weight_decays[i+1]], 
			color=phase_colors[phase], 
			linewidth=2.5, 
			alpha=0.8
		)
	
	# Mark phase transitions
	for transition_epoch in phase_transitions:
		ax3.axvline(
			x=transition_epoch, 
			color='orange', 
			linestyle='--', 
			linewidth=1.5, 
			alpha=0.7,
		)
	
	ax3.set_xlabel('Epoch', fontsize=11, weight='bold')
	ax3.set_ylabel('Weight Decay', fontsize=11, weight='bold')
	ax3.set_title('Weight Decay Adaptation', fontsize=11, weight='bold')
	ax3.set_yscale('log')
	ax3.grid(True, alpha=0.3)
	
	# ================================
	# 4. Trainable Parameters Evolution (third row, left)
	# ================================
	ax4 = fig.add_subplot(gs[2, 0])
	
	# Calculate trainable parameters per phase
	trainable_params_per_phase = []
	total_params = sum(len(layers) for layers in layer_groups.values())
	
	for phase in range(max(phases) + 1):
		if phase in unfreeze_schedule:
			unfrozen_layers = len(unfreeze_schedule[phase])
			trainable_params_per_phase.append(unfrozen_layers)
		else:
			trainable_params_per_phase.append(0)
	
	# Create step plot for parameter evolution
	phase_epochs_dict = {}
	for epoch, phase in zip(epochs, phases):
		if phase not in phase_epochs_dict:
			phase_epochs_dict[phase] = []
		phase_epochs_dict[phase].append(epoch)
	
	for phase in sorted(phase_epochs_dict.keys()):
		phase_epochs_list = phase_epochs_dict[phase]
		param_count = trainable_params_per_phase[phase] if phase < len(trainable_params_per_phase) else 0
		ax4.hlines(
			param_count, 
			min(phase_epochs_list), 
			max(phase_epochs_list), 
			colors=phase_colors[phase], 
			linewidth=4, 
			alpha=0.8,
			label=f'Phase {phase}'
		)
	
	# Mark phase transitions
	for transition_epoch in phase_transitions:
		ax4.axvline(
			x=transition_epoch, 
			color='orange', 
			linestyle='--', 
			linewidth=1.5, 
			alpha=0.7
		)
	
	ax4.set_xlabel('Epoch', fontsize=11, weight='bold')
	ax4.set_ylabel('Trainable Layers', fontsize=11, weight='bold')
	ax4.set_title('Layer Unfreezing Progression', fontsize=12, weight='bold')
	ax4.set_ylim(0, total_params * 1.1)
	ax4.grid(True, alpha=0.3)
	
	# ================================
	# 5. Layer Group Unfreezing Heatmap (third row, right)
	# ================================
	ax5 = fig.add_subplot(gs[2, 1])
	
	# Create heatmap data: rows = layer groups, columns = phases
	group_names = list(layer_groups.keys())
	num_phases = max(phases) + 1
	heatmap_data = np.zeros((len(group_names), num_phases))
	
	for phase_idx in range(num_phases):
		if phase_idx in unfreeze_schedule:
			unfrozen_layers = set(unfreeze_schedule[phase_idx])
			for group_idx, (group_name, group_layers) in enumerate(layer_groups.items()):
				# Calculate percentage of group layers unfrozen
				group_unfrozen = len([layer for layer in group_layers if any(ul in layer for ul in unfrozen_layers)])
				group_total = len(group_layers)
				heatmap_data[group_idx, phase_idx] = group_unfrozen / max(group_total, 1)
	
	# Create custom colormap
	colors = ['#f0f0f0', '#d0e7ff', "#4190eb"]
	n_bins = 100
	cmap = LinearSegmentedColormap.from_list('unfreezing', colors, N=n_bins)
	
	im = ax5.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=0, vmax=1)
	
	# Set ticks and labels
	ax5.set_xticks(range(num_phases))
	ax5.set_xticklabels([f'P{i}' for i in range(num_phases)], fontsize=8)
	ax5.set_yticks(range(len(group_names)))
	ax5.set_yticklabels([name.replace('_', '\n').title() for name in group_names], fontsize=8)
	
	ax5.set_xlabel('Phase', fontsize=8, weight='bold')
	ax5.set_ylabel('Layer Groups', fontsize=8, weight='bold')
	ax5.set_title('Layer Group Unfreezing Pattern', fontsize=10, weight='bold')
	
	# Add colorbar
	cbar = plt.colorbar(im, ax=ax5, shrink=0.8)
	cbar.set_label('Fraction Unfrozen', fontsize=8)
	
	# ================================
	# 6. Training Statistics Summary
	# ================================
	
	# Calculate summary statistics
	total_epochs = len(epochs)
	num_phases = len(set(phases))
	final_train_loss = train_losses[-1] if train_losses else 0
	final_val_loss = val_losses[-1] if val_losses else 0
	best_val_loss = min(val_losses) if val_losses else 0
	improvement = ((val_losses[0] - best_val_loss) / val_losses[0] * 100) if val_losses and val_losses[0] > 0 else 0
	
	# Create summary text
	summary_text = f"""
	TRAINING SUMMARY:
	• Total Epochs: {total_epochs}
	• Number of Phases: {num_phases}
	• Phase Transitions: {len(phase_transitions)}
	• Final Training Loss: {final_train_loss:.6f}
	• Final Validation Loss: {final_val_loss:.6f}
	• Best Validation Loss: {best_val_loss:.6f}
	• Total Improvement: {improvement:.2f}%
	"""
	
	if early_stop_epoch:
		summary_text += f"    • Early Stopped at Epoch: {early_stop_epoch}\n"
	if best_epoch is not None:
		summary_text += f"    • Best Model at Epoch: {epochs[best_epoch]}\n"
	
	# Add layer unfreezing summary
	summary_text += "\n    LAYER UNFREEZING SCHEDULE:\n"
	for phase, layers in unfreeze_schedule.items():
		summary_text += f"    • Phase {phase}: {len(layers)} layers unfrozen\n"
			
	print(f"Training summary:\n{summary_text}\n")
	
	plt.suptitle(f'Progressive Fine-tuning Dynamics',fontsize=12, weight='bold')	
	plt.savefig(
		save_path, 
		dpi=300, 
		bbox_inches='tight', 
		facecolor='white', 
		edgecolor='none'
	)
	plt.close()

def collect_progressive_training_history(
		training_losses: List[float],
		in_batch_metrics_all_epochs: List[Dict],
		learning_rates: List[float],
		weight_decays: List[float],
		phases: List[int],
		phase_transitions: List[int],
		early_stop_epoch: Optional[int] = None,
		best_epoch: Optional[int] = None
	) -> Dict:
	"""
	Collect training history data for progressive training visualization.
	
	This function should be called at the end of progressive training to gather
	all the necessary data for visualization.
	
	Args:
		training_losses: List of training losses per epoch
		in_batch_metrics_all_epochs: List of validation metrics per epoch
		learning_rates: List of learning rates per epoch
		weight_decays: List of weight decays per epoch  
		phases: List of phase numbers per epoch
		phase_transitions: List of epochs where phase transitions occurred
		early_stop_epoch: Epoch where early stopping occurred (if any)
		best_epoch: Epoch with best validation performance
	
	Returns:
		Dictionary containing all training history data
	"""
	
	val_losses = [metrics.get('val_loss', 0.0) for metrics in in_batch_metrics_all_epochs]
	epochs = list(range(len(training_losses)))
	
	return {
		'epochs': epochs,
		'train_losses': training_losses,
		'val_losses': val_losses,
		'learning_rates': learning_rates,
		'weight_decays': weight_decays,
		'phases': phases,
		'phase_transitions': phase_transitions,
		'early_stop_epoch': early_stop_epoch,
		'best_epoch': best_epoch
	}

def plot_multilabel_loss_breakdown(
		training_losses_breakdown: Dict[str, List[float]],
		filepath: str,
		figure_size=(12, 8),
		DPI: int = 300,
	):
	print(f"Plotting multi-label loss breakdown to {filepath}...")
	print(training_losses_breakdown)
	# --- Debug: Print loss lengths for verification ---
	print("\n[Plotting] Loss list lengths before plotting:")
	for key, loss_list in training_losses_breakdown.items():
		if isinstance(loss_list, list):
			print(f"  - {key:<10}: {len(loss_list)} values")
		else:
			print(f"  - {key:<10}: Not a list (type={type(loss_list).__name__})")

	# Find the number of epochs from the longest valid list in the dictionary
	num_epochs = 0
	if training_losses_breakdown:
		valid_lists = [v for v in training_losses_breakdown.values() if isinstance(v, list) and v]
		if valid_lists:
			num_epochs = max(len(v) for v in valid_lists)

	if num_epochs == 0:
		print("[Warning] No valid loss data to plot. Skipping plot generation.")
		return

	epochs = range(1, num_epochs + 1)
	plt.figure(figsize=figure_size)

	# Define plotting styles for known loss components for consistency
	styles = {
		"total": {'color': 'b', 'linestyle': '-', 'linewidth': 1.5, 'label': 'Total Loss'},
		"i2t": {'color': 'g', 'linestyle': '--', 'linewidth': 2.0, 'label': 'Image→Text Loss'},
		"t2i": {'color': 'r', 'linestyle': '--', 'linewidth': 2.0, 'label': 'Text→Image Loss'},
	}

	# Plot each loss component present in the dictionary
	for key, loss_list in training_losses_breakdown.items():
		if isinstance(loss_list, list) and len(loss_list) > 0:
			# Pad with NaN if a list is shorter than the max number of epochs
			padded_list = loss_list + [float('nan')] * (num_epochs - len(loss_list))

			# Get style from the dictionary or use a default
			plot_kwargs = styles.get(key, {'label': key.replace('_', ' ').title()})
			plt.plot(epochs, padded_list, **plot_kwargs)

	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Training Loss Breakdown')
	plt.legend(title='Loss Components', fontsize=10, title_fontsize=12, loc='upper right')
	plt.grid(True, which='both', linestyle='--', linewidth=0.5)
	plt.tight_layout()

	try:
		plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
		print(f"Successfully saved loss breakdown plot to {filepath}")
	except Exception as e:
		print(f"Error saving plot to {filepath}: {e}")
	finally:
		plt.close()

def plot_image_to_texts_separate_horizontal_bars(
				models: dict,
				validation_loader: DataLoader,
				preprocess,
				img_path: str,
				topk: int,
				device: str,
				results_dir: str,
				dpi: int = 250,
		):
		dataset_name = getattr(validation_loader, 'name', 'unknown_dataset')
		pretrained_model_arch = models.get("pretrained").name
		print(f"{len(models)} strategies for {dataset_name} {pretrained_model_arch}")
		
		# Prepare labels
		try:
				labels = validation_loader.dataset.dataset.classes
		except AttributeError:
				labels = validation_loader.dataset.unique_labels
		n_labels = len(labels)
		if topk > n_labels:
				print(f"ERROR: requested Top-{topk} labeling is greater than number of labels ({n_labels}) => EXIT...")
				return
		tokenized_labels_tensor = clip.tokenize(texts=labels).to(device)
		
		# Load and preprocess image
		try:
				img = Image.open(img_path).convert("RGB")
		except FileNotFoundError:
				try:
						response = requests.get(img_path)
						response.raise_for_status()
						img = Image.open(BytesIO(response.content)).convert("RGB")
				except requests.exceptions.RequestException as e:
						print(f"ERROR: failed to load image from {img_path} => {e}")
						return
		image_tensor = preprocess(img).unsqueeze(0).to(device)
		
		# Check if img_path is in the validation set and get ground-truth label if available
		ground_truth_label = None
		validation_dataset = validation_loader.dataset
		if hasattr(validation_dataset, 'data_frame') and 'img_path' in validation_dataset.data_frame.columns:
				matching_rows = validation_dataset.data_frame[validation_dataset.data_frame['img_path'] == img_path]
				if not matching_rows.empty:
						ground_truth_label = matching_rows['label'].iloc[0]
						print(f"Ground truth label for {img_path}: {ground_truth_label}")
		
		# Compute predictions for each model
		model_predictions = {}
		model_topk_labels = {}
		model_topk_probs = {}
		for model_name, model in models.items():
				model.eval()
				print(f"[Image-to-text(s)] {model_name} Zero-Shot Image Classification Query: {img_path}".center(200, " "))
				t0 = time.time()
				with torch.no_grad():
						image_features = model.encode_image(image_tensor)
						labels_features = model.encode_text(tokenized_labels_tensor)
						image_features /= image_features.norm(dim=-1, keepdim=True)
						labels_features /= labels_features.norm(dim=-1, keepdim=True)
						similarities = (100.0 * image_features @ labels_features.T).softmax(dim=-1)
				
				# Store full probabilities for all labels
				all_probs = similarities.squeeze().cpu().numpy()
				model_predictions[model_name] = all_probs
				
				# Get top-k labels and probabilities for this model
				topk_pred_probs, topk_pred_labels_idx = similarities.topk(topk, dim=-1)
				topk_pred_probs = topk_pred_probs.squeeze().cpu().numpy()
				topk_pred_indices = topk_pred_labels_idx.squeeze().cpu().numpy()
				topk_pred_labels = [labels[i] for i in topk_pred_indices]
				
				# Sort by descending probability
				sorted_indices = np.argsort(topk_pred_probs)[::-1]
				model_topk_labels[model_name] = [topk_pred_labels[i] for i in sorted_indices]
				model_topk_probs[model_name] = topk_pred_probs[sorted_indices]
				print(f"Top-{topk} predicted labels for {model_name}: {model_topk_labels[model_name]}")
				print(f"Elapsed_t: {time.time()-t0:.3f} sec".center(160, "-"))
		
		# IMPROVED LAYOUT CALCULATION
		# Get image dimensions for dynamic sizing
		img_width, img_height = img.size
		aspect_ratio = img_height / img_width
		
		# Number of models to display
		num_strategies = len(models)
		
		# Base the entire layout on the image aspect ratio
		img_display_width = 4  # Base width for image in inches
		img_display_height = img_display_width * aspect_ratio
		
		# Set model result panels to have identical height as the image
		# Each model panel should have a fixed width ratio relative to the image
		model_panel_width = 3.5  # Width for each model panel
		
		# Calculate total figure dimensions
		fig_width = img_display_width + (model_panel_width * num_strategies)
		fig_height = max(4, img_display_height)  # Ensure minimum height
		
		# Create figure
		fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
		
		# Create grid with precise width ratios
		# First column for image, remaining columns for models
		width_ratios = [img_display_width] + [model_panel_width] * num_strategies
		
		# Create GridSpec with exact dimensions
		gs = gridspec.GridSpec(
				1, 
				1 + num_strategies, 
				width_ratios=width_ratios,
				wspace=0.05  # Reduced whitespace between panels
		)
		
		# Subplot 1: Query Image
		ax0 = fig.add_subplot(gs[0])
		ax0.imshow(img)
		ax0.axis('off')
		
		# Add title with ground truth if available
		title_text = f"Query Image\nGT: {ground_truth_label.capitalize()}" if ground_truth_label else "Query Image"
		ax0.text(
				0.5,  # x position (center)
				-0.05,  # y position (just below the image)
				title_text,
				fontsize=10,
				fontweight='bold',
				ha='center',
				va='top',
				transform=ax0.transAxes  # Use axes coordinates
		)
		
		# Define colors consistent with plot_comparison_metrics_split/merged
		strategy_colors = {'full': '#0058a5', 'lora': '#f58320be', 'progressive': '#cc40df'}  # Blue, Orange, Purple
		pretrained_colors = {'ViT-B/32': '#745555', 'ViT-B/16': '#9467bd', 'ViT-L/14': '#e377c2', 'ViT-L/14@336px': '#696969'}
		colors = [pretrained_colors.get(pretrained_model_arch, '#000000')] + list(strategy_colors.values())
		print(f"colors: {colors}")
		
		# Subplots for each model
		all_strategies = list(models.keys())
		axes = []
		
		# Create subplots for models - ensuring dimensions are consistent
		for model_idx in range(num_strategies):
				ax = fig.add_subplot(gs[model_idx + 1])
				axes.append(ax)
		
		# Create a list of handles for the legend
		legend_handles = []
		
		# Plot data for each model
		for model_idx, (model_name, ax) in enumerate(zip(all_strategies, axes)):
				y_pos = np.arange(topk)
				sorted_probs = model_topk_probs[model_name]
				sorted_labels = model_topk_labels[model_name]
				
				# Plot horizontal bars and create a handle for the legend
				bars = ax.barh(
						y_pos,
						sorted_probs,
						height=0.5,
						color=colors[model_idx],
						edgecolor='white',
						alpha=0.9,
						label=f"CLIP {pretrained_model_arch}" if model_name == "pretrained" else model_name.upper()
				)
				legend_handles.append(bars)
				
				# Format axis appearance
				ax.invert_yaxis()  # Highest probs on top
				ax.set_yticks([])  # Hide y-axis ticks 
				ax.set_yticklabels([])  # Empty labels
				
				# Set consistent x-axis limits and ticks
				ax.set_xlim(0, 1)
				ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
				ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1.0'], fontsize=8)
				ax.grid(True, axis='x', linestyle='--', alpha=0.5, color='#888888')
				
				# Annotate bars with labels and probabilities
				for i, (label, prob) in enumerate(zip(sorted_labels, sorted_probs)):
						formatted_label = label.replace('_', ' ').title()
						ax.text(
								prob + 0.01 if prob < 0.5 else prob - 0.01,
								i,
								f"{formatted_label}\n({prob:.2f})",
								va='center',
								ha='right' if prob > 0.5 else 'left',
								fontsize=8,
								color='black',
								fontweight='bold' if prob == max(sorted_probs) else 'normal',
						)
				
				# Set border color
				for spine in ax.spines.values():
						spine.set_color('black')
		
		# Add a legend at the top of the figure
		fig.legend(
				legend_handles,
				[handle.get_label() for handle in legend_handles],
				fontsize=11,
				loc='upper center',
				ncol=len(legend_handles),
				bbox_to_anchor=(0.5, 1.02),
				bbox_transform=fig.transFigure,
				frameon=True,
				shadow=True,
				fancybox=True,
				edgecolor='black',
				facecolor='white',
		)
		
		# Add x-axis label
		fig.text(
				0.5,  # x position (center of figure)
				0.02,  # y position (near bottom of figure)
				"Probability",
				ha='center',
				va='center',
				fontsize=12,
				fontweight='bold'
		)
		
		# IMPORTANT: Instead of tight_layout which can override our settings,
		# use a more controlled approach
		fig.subplots_adjust(top=0.85, bottom=0.1, left=0.05, right=0.95)
		
		# Save the figure
		img_hash = hashlib.sha256(img_path.encode()).hexdigest()[:8]
		file_name = os.path.join(
				results_dir,
				f'{dataset_name}_'
				f'Top{topk}_labels_'
				f'image_{img_hash}_'
				f"{'gt_' + ground_truth_label.replace(' ', '-') + '_' if ground_truth_label else ''}"
				f"{re.sub(r'[/@]', '-', pretrained_model_arch)}_"
				f'separate_bar_image_to_text.png'
		)
		
		plt.savefig(file_name, bbox_inches='tight', dpi=dpi)
		plt.close()
		print(f"Saved visualization to: {file_name}")

def plot_image_to_texts_stacked_horizontal_bar(
		models: dict,
		validation_loader: DataLoader,
		preprocess,
		img_path: str,
		topk: int,
		device: str,
		results_dir: str,
		figure_size=(8, 6),
		dpi: int = 250,
	):
	dataset_name = getattr(validation_loader, 'name', 'unknown_dataset')
	print(f"num_strategies: {len(models)}")
	try:
		labels = validation_loader.dataset.dataset.classes
	except AttributeError:
		labels = validation_loader.dataset.unique_labels
	n_labels = len(labels)
	if topk > n_labels:
		print(f"ERROR: requested Top-{topk} labeling is greater than number of labels ({n_labels}) => EXIT...")
		return
	tokenized_labels_tensor = clip.tokenize(texts=labels).to(device)
	try:
		img = Image.open(img_path).convert("RGB")
	except FileNotFoundError:
		try:
			response = requests.get(img_path)
			response.raise_for_status()
			img = Image.open(BytesIO(response.content)).convert("RGB")
		except requests.exceptions.RequestException as e:
			print(f"ERROR: failed to load image from {img_path} => {e}")
			return
	image_tensor = preprocess(img).unsqueeze(0).to(device)

	# Compute predictions for each model
	model_predictions = {}
	pretrained_topk_labels = []  # To store the top-k labels from the pre-trained model
	pretrained_topk_probs = []  # To store the corresponding probabilities for sorting
	for model_name, model in models.items():
		model.eval()
		print(f"[Image-to-text(s)] {model_name} Zero-Shot Image Classification of image: {img_path}".center(200, " "))
		t0 = time.time()
		with torch.no_grad():
			image_features = model.encode_image(image_tensor)
			labels_features = model.encode_text(tokenized_labels_tensor)
			image_features /= image_features.norm(dim=-1, keepdim=True)
			labels_features /= labels_features.norm(dim=-1, keepdim=True)
			similarities = (100.0 * image_features @ labels_features.T).softmax(dim=-1)
		
		# Store full probabilities for all labels
		all_probs = similarities.squeeze().cpu().numpy()
		model_predictions[model_name] = all_probs
		# If this is the pre-trained model, get its top-k labels and probabilities
		if model_name == "pretrained":
			topk_pred_probs, topk_pred_labels_idx = similarities.topk(topk, dim=-1)
			topk_pred_probs = topk_pred_probs.squeeze().cpu().numpy()
			topk_pred_indices = topk_pred_labels_idx.squeeze().cpu().numpy()
			pretrained_topk_labels = [labels[i] for i in topk_pred_indices]
			pretrained_topk_probs = topk_pred_probs
			print(f"Top-{topk} predicted labels for pretrained model: {pretrained_topk_labels}")
		print(f"Elapsed_t: {time.time()-t0:.3f} sec".center(160, "-"))

	# Sort the pre-trained model's top-k labels by their probabilities (descending)
	sorted_indices = np.argsort(pretrained_topk_probs)[::-1]  # Descending order
	pretrained_topk_labels = [pretrained_topk_labels[i] for i in sorted_indices]

	num_labels = len(pretrained_topk_labels)
	num_strategies = len(models)
	plot_data = np.zeros((num_labels, num_strategies))  # Rows: labels, Columns: models
	all_strategies = list(models.keys())

	for model_idx, (model_name, probs) in enumerate(model_predictions.items()):
		for label_idx, label in enumerate(pretrained_topk_labels):
			# Find the index of this label in the full label list
			label_full_idx = labels.index(label)
			plot_data[label_idx, model_idx] = probs[label_full_idx]

	fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)
	bar_width = 0.21
	y_pos = np.arange(num_labels)
	strategy_colors = {'full': '#0058a5', 'lora': '#f58320be', 'progressive': '#cc40df'}  # Blue, Orange, Green
	pretrained_colors = {'ViT-B/32': '#745555', 'ViT-B/16': '#9467bd', 'ViT-L/14': '#e377c2', 'ViT-L/14@336px': '#696969'}
	pretrained_model_arch = models.get("pretrained").name
	colors = [pretrained_colors.get(pretrained_model_arch, '#000000')] + list(strategy_colors.values())
	print(f"colors: {colors}")
	winning_model_per_label = np.argmax(plot_data, axis=1)

	for model_idx, model_name in enumerate(all_strategies):
			if model_name == "pretrained":
				model_name = f"{model_name.capitalize()} {pretrained_model_arch}"
			offset = (model_idx - num_strategies / 2) * bar_width
			bars = ax.barh(
				y_pos + offset,
				plot_data[:, model_idx],
				height=bar_width,
				label=model_name.split('_')[-1].replace('finetune', '').capitalize() if '_' in model_name else model_name,
				color=colors[model_idx],
				edgecolor='white',
				alpha=0.85,
			)
			for i, bar in enumerate(bars):
				width = bar.get_width()
				if width > 0.01:
					is_winner = (model_idx == winning_model_per_label[i])
					ax.text(
						width + 0.01,
						bar.get_y() + bar.get_height() / 2,
						f"{width:.2f}",
						va='center',
						fontsize=8,
						color='black',
						fontweight='bold' if is_winner else 'normal',
						alpha=0.85,
					)
	ax.set_yticks(y_pos)
	ax.set_yticklabels([label.replace('_', ' ').title() for label in pretrained_topk_labels], fontsize=11)
	ax.set_xlim(0, 1.02)
	ax.set_xlabel("Probability", fontsize=10)
	# ax.set_title(f"Top-{topk} Predictions (Pre-trained Baseline)", fontsize=12, fontweight='bold')
	ax.grid(True, axis='x', linestyle='--', alpha=0.5, color='black',)
	ax.tick_params(axis='x', labelsize=12)
	ax.legend(
		fontsize=9,
		loc='best',
		ncol=len(models),
		frameon=True,
		facecolor='white',
		shadow=True,
		fancybox=True,
	)
	for spine in ax.spines.values():
		spine.set_color('black')
	img_hash = hashlib.sha256(img_path.encode()).hexdigest()[:8]
	file_name = os.path.join(
			results_dir,
			f'{dataset_name}_'
			f'Top{topk}_labels_'
			f'image_{img_hash}_'
			f"{re.sub(r'[/@]', '-', pretrained_model_arch)}_"
			f'stacked_bar_image_to_text.png'
	)
	plt.tight_layout()
	plt.savefig(file_name, bbox_inches='tight', dpi=dpi)
	plt.close()
	print(f"Saved visualization to: {file_name}")

	# Save the original image separately using same hash (for visual comparison)
	fig_img, ax_img = plt.subplots(figsize=(4, 4), dpi=dpi)
	ax_img.imshow(img)
	ax_img.axis('off')
	img_file_name = os.path.join(results_dir, f'{dataset_name}_query_original_image_{img_hash}.png')
	plt.tight_layout()
	plt.savefig(img_file_name, bbox_inches='tight', dpi=dpi)
	plt.close()
	print(f"Saved original image to: {img_file_name}")

def plot_image_to_texts_pretrained(
		best_pretrained_model: torch.nn.Module,
		validation_loader: DataLoader,
		preprocess,
		img_path: str,
		topk: int,
		device: str,
		results_dir: str,
		figure_size=(13, 7),
		dpi: int = 300,
	):
	dataset_name = getattr(validation_loader, 'name', 'unknown_dataset')
	best_pretrained_model_name = best_pretrained_model.__class__.__name__
	best_pretrained_model_arch = re.sub(r'[/@]', '-', best_pretrained_model.name)
	best_pretrained_model.eval()
	print(f"[Image-to-text(s)] {best_pretrained_model_name} {best_pretrained_model_arch} Zero-Shot Image Classification of image: {img_path}".center(200, " "))
	t0 = time.time()
	try:
		labels = validation_loader.dataset.dataset.classes
	except AttributeError:
		labels = validation_loader.dataset.unique_labels
	n_labels = len(labels)
	if topk > n_labels:
		print(f"ERROR: requested Top-{topk} labeling is greater than number of labels ({n_labels}) => EXIT...")
		return
	# Tokenize the labels and move to device
	tokenized_labels_tensor = clip.tokenize(texts=labels).to(device)

	try:
		img = Image.open(img_path).convert("RGB")
	except FileNotFoundError:
		try:
			response = requests.get(img_path)
			response.raise_for_status()
			img = Image.open(BytesIO(response.content)).convert("RGB")
		except requests.exceptions.RequestException as e:
			print(f"ERROR: failed to load image from {img_path} => {e}")
			return

	# Preprocess image
	image_tensor = preprocess(img).unsqueeze(0).to(device)
	
	# Encode and compute similarity
	with torch.no_grad():
		image_features = best_pretrained_model.encode_image(image_tensor)
		labels_features = best_pretrained_model.encode_text(tokenized_labels_tensor)
		image_features /= image_features.norm(dim=-1, keepdim=True)
		labels_features /= labels_features.norm(dim=-1, keepdim=True)
		similarities = (100.0 * image_features @ labels_features.T).softmax(dim=-1)
	
	# Get top-k predictions
	topk_pred_probs, topk_pred_labels_idx = similarities.topk(topk, dim=-1)
	topk_pred_probs = topk_pred_probs.squeeze().cpu().numpy()
	topk_pred_indices = topk_pred_labels_idx.squeeze().cpu().numpy()
	topk_pred_labels = [labels[i] for i in topk_pred_indices]
	print(f"Top-{topk} predicted labels: {topk_pred_labels}")

	# Sort predictions by descending probability
	sorted_indices = topk_pred_probs.argsort()[::-1]
	sorted_probs = topk_pred_probs[sorted_indices]
	print(sorted_probs)

	sorted_labels = [topk_pred_labels[i] for i in sorted_indices]

	# Hash image path for unique filename
	img_hash = hashlib.sha256(img_path.encode()).hexdigest()[:8]
	file_name = os.path.join(
			results_dir,
			f'{dataset_name}_'
			f'Top{topk}_labels_'
			f'image_{img_hash}_'
			f"{re.sub(r'[/@]', '-', best_pretrained_model_arch)}_pretrained_"
			f'bar_image_to_text.png'
	)
	strategy_colors = {'full': '#0058a5', 'lora': '#f58320be', 'progressive': '#cc40df'}  # Blue, Orange, Green
	pretrained_colors = {'ViT-B/32': '#745555', 'ViT-B/16': '#9467bd', 'ViT-L/14': '#e377c2', 'ViT-L/14@336px': '#696969'}

	# Plot
	fig = plt.figure(figsize=figure_size, dpi=dpi)
	gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.05], wspace=0.01)
	# Subplot 1: Image
	ax0 = plt.subplot(gs[0])
	ax0.imshow(img)
	ax0.axis('off')
	ax0.set_title("Query Image", fontsize=12)

	# Subplot 2: Horizontal bar plot
	ax1 = plt.subplot(gs[1])
	y_pos = range(topk)
	ax1.barh(y_pos, sorted_probs, color=pretrained_colors.get(best_pretrained_model.name, '#000000'), edgecolor='white')
	ax1.invert_yaxis()  # Highest probs on top
	ax1.set_yticks([])  # Hide y-axis ticks
	ax1.set_xlim(0, 1)
	ax1.set_xlabel("Probability", fontsize=11)
	ax1.set_title(f"Top-{topk} Predicted Labels", fontsize=10)
	ax1.grid(False)
	# ax1.grid(True, axis='x', linestyle='--', alpha=0.5, color='black')
	for spine in ax1.spines.values():
		spine.set_edgecolor('black')

	# Annotate bars on the right with labels and probs
	for i, (label, prob) in enumerate(zip(sorted_labels, sorted_probs)):
		ax1.text(prob + 0.02, i, f"{label} ({prob:.2f})", va='center', fontsize=8, color='black', fontweight='bold', backgroundcolor='white', alpha=0.8)
	
	plt.tight_layout()
	plt.savefig(file_name, bbox_inches='tight')
	plt.close()
	print(f"Saved visualization to: {file_name}")
	print(f"Elapsed_t: {time.time()-t0:.3f} sec".center(160, "-"))

def plot_text_to_images_merged(
		models: Dict,
		validation_loader: torch.utils.data.DataLoader,
		preprocess,
		query_text: str,
		topk: int,
		device: str,
		results_dir: str,
		cache_dir: str = None,
		embeddings_cache: Dict = None,
		dpi: int = 300,
		print_every: int = 250
	):
	dataset_name = getattr(validation_loader, 'name', 'unknown_dataset')
	img_hash = hashlib.sha256(query_text.encode()).hexdigest()[:8]
	if cache_dir is None:
		cache_dir = results_dir
	
	strategy_colors = {'full': '#0058a5', 'lora': '#f58320be', 'progressive': '#cc40df'}
	pretrained_colors = {'ViT-B/32': '#745555', 'ViT-B/16': '#9467bd', 'ViT-L/14': '#e377c2', 'ViT-L/14@336px': '#696969'}
	
	pretrained_model_arch = models.get("pretrained").name
	tokenized_query = clip.tokenize([query_text]).to(device)
	
	model_results = {}
	num_strategies = len(models)
	all_strategies = list(models.keys())
	
	for strategy, model in models.items():
			print(f"Processing strategy: {strategy} {model.__class__.__name__} {pretrained_model_arch}".center(160, " "))
			model.eval()
			print(f"[Text-to-image(s) (merged)] {strategy} query: '{query_text}'".center(160, " "))
			
			# Use cached embeddings
			if embeddings_cache is not None and strategy in embeddings_cache:
					all_image_embeddings, image_paths = embeddings_cache[strategy]
					print(f"Using precomputed embeddings for {strategy}")
			else:
					print(f"No cached embeddings found for {strategy}. Computing from scratch...")
					cache_file = os.path.join(
							cache_dir, 
							f"{dataset_name}_{strategy}_{model.__class__.__name__}_{re.sub(r'[/@]', '-', pretrained_model_arch)}_embeddings.pt"
					)
					
					all_image_embeddings = None
					image_paths = []
					
					if os.path.exists(cache_file):
							print(f"Loading cached embeddings from {cache_file}")
							try:
									cached_data = torch.load(cache_file, map_location='cpu')
									all_image_embeddings = cached_data['embeddings'].to(device, dtype=torch.float32)
									image_paths = cached_data.get('image_paths', [])
									print(f"Successfully loaded {len(all_image_embeddings)} cached embeddings")
							except Exception as e:
									print(f"Error loading cached embeddings: {e}")
									all_image_embeddings = None
					
					if all_image_embeddings is None:
							print("Computing image embeddings (this may take a while)...")
							image_embeddings_list = []
							image_paths = []
							
							dataset = validation_loader.dataset
							has_img_path = hasattr(dataset, 'images') and isinstance(dataset.images, (list, tuple))
							for batch_idx, batch in enumerate(validation_loader):
									images = batch[0]
									if has_img_path:
											start_idx = batch_idx * validation_loader.batch_size
											batch_paths = []
											for i in range(len(images)):
													global_idx = start_idx + i
													if global_idx < len(dataset):
															batch_paths.append(dataset.images[global_idx])
													else:
															batch_paths.append(f"missing_path_{global_idx}")
									else:
											batch_paths = [f"batch_{batch_idx}_img_{i}" for i in range(len(images))]
									
									image_paths.extend(batch_paths)
									
									if not isinstance(images, torch.Tensor) or len(images.shape) != 4:
											print(f"Warning: Invalid image tensor in batch {batch_idx}")
											continue
									
									with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=True):
											images = images.to(device)
											image_features = model.encode_image(images)
											image_features /= image_features.norm(dim=-1, keepdim=True)
											image_embeddings_list.append(image_features.cpu().to(torch.float32))
									
									if (batch_idx + 1) % print_every == 0:
											print(f"Processed {batch_idx + 1}/{len(validation_loader)} batches")
							
							if image_embeddings_list:
									all_image_embeddings = torch.cat(image_embeddings_list, dim=0).to(device, dtype=torch.float32)
									print(f"Computed {len(all_image_embeddings)} image embeddings for {dataset_name} | « {strategy} » strategy")
									
									try:
											torch.save({
													'embeddings': all_image_embeddings.cpu(),
													'image_paths': image_paths
											}, cache_file)
											print(f"Saved embeddings to {cache_file}")
									except Exception as e:
											print(f"Warning: Failed to save embeddings cache: {e}")
							else:
									print("Error: No valid image embeddings were collected")
									continue
			
			# Compute similarities between text query and all images
			with torch.no_grad():
					text_features = model.encode_text(tokenized_query)
					text_features = F.normalize(text_features, dim=-1).to(torch.float32)
					all_image_embeddings = all_image_embeddings.to(torch.float32)
					similarities = (100.0 * text_features @ all_image_embeddings.T).softmax(dim=-1)
					
					effective_topk = min(topk, len(all_image_embeddings))
					topk_scores, topk_indices = torch.topk(similarities.squeeze(), effective_topk)
					topk_scores = topk_scores.cpu().numpy()
					topk_indices = topk_indices.cpu().numpy()
			
			# Retrieve ground-truth labels from the dataset
			dataset = validation_loader.dataset
			try:
				if hasattr(dataset, 'label') and isinstance(dataset.label, (list, np.ndarray)):
					ground_truth_labels = dataset.label
				elif hasattr(dataset, 'labels') and isinstance(dataset.labels, (list, np.ndarray)):
					ground_truth_labels = dataset.labels
				else:
					raise AttributeError("Dataset does not have accessible 'label' or 'labels' attribute")
				topk_ground_truth_labels = [ground_truth_labels[idx] for idx in topk_indices]
			except (AttributeError, IndexError) as e:
				print(f"Warning: Could not retrieve ground-truth labels: {e}")
				topk_ground_truth_labels = [f"Unknown GT {idx}" for idx in topk_indices]
			
			# Store results for this model
			model_results[strategy] = {
				'topk_scores': topk_scores,
				'topk_indices': topk_indices,
				'image_paths': image_paths,
				'ground_truth_labels': topk_ground_truth_labels
			}
	
	# Create a figure with a larger figure size to accommodate the borders
	fig_width = effective_topk * 7
	fig_height = num_strategies * 4.5
	fig, axes = plt.subplots(
			nrows=num_strategies,
			ncols=effective_topk,
			figsize=(fig_width, fig_height),
			constrained_layout=True,
	)
	fig.suptitle(
			f"Query: '{query_text}' Top-{effective_topk} Images Across Models",
			fontsize=13,
			fontweight='bold',
	)
	
	# If there's only one model or topk=1, adjust axes to be 2D
	if num_strategies == 1:
			axes = [axes]
	if effective_topk == 1:
			axes = [[ax] for ax in axes]
	
	# Plot images for each model
	for row_idx, strategy in enumerate(all_strategies):
			# Get border color for this model
			if strategy == 'pretrained':
					model = models[strategy]
					border_color = pretrained_colors.get(model.name, '#745555')
			else:
					border_color = strategy_colors.get(strategy, '#000000')
			
			# Get top-k results for this model
			result = model_results[strategy]
			topk_scores = result['topk_scores']
			topk_indices = result['topk_indices']
			image_paths = result['image_paths']
			topk_ground_truth_labels = result['ground_truth_labels']
			dataset = validation_loader.dataset
			
			# Plot each image in the row
			for col_idx, (idx, score, gt_label) in enumerate(zip(topk_indices, topk_scores, topk_ground_truth_labels)):
				ax = axes[row_idx][col_idx]
				try:
					img_path = image_paths[idx]
					if os.path.exists(img_path):
						img = Image.open(img_path).convert('RGB')
						ax.imshow(img)
					else:
						if hasattr(dataset, '__getitem__'):
							sample = dataset[idx]
							if len(sample) >= 3:
								img = sample[0]
							else:
								raise ValueError(f"Unexpected dataset structure at index {idx}: {sample}")
							if isinstance(img, torch.Tensor):
								img = img.cpu().numpy()
								if img.shape[0] in [1, 3]:
									img = img.transpose(1, 2, 0)
								mean = np.array([0.5126933455467224, 0.5045100450515747, 0.48094621300697327])
								std = np.array([0.276103675365448, 0.2733437418937683, 0.27065524458885193])
								img = img * std + mean
								img = np.clip(img, 0, 1)
							ax.imshow(img)
					ax.set_title(f"Top-{col_idx+1} (Score: {score:.4f})\nGT: {gt_label}", fontsize=9)
				except Exception as e:
					print(f"Warning: Could not display image {idx} for model {strategy}: {e}")
					ax.imshow(np.ones((224, 224, 3)) * 0.5)
					ax.set_title(f"Top-{col_idx+1} (Score: {score:.4f})\nGT: Unknown", fontsize=10)
				
				# Remove default spines
				for spine in ax.spines.values():
					spine.set_visible(False)
				
				ax.axis('off')
			
			# Add model name label on the left side of the row
			axes[row_idx][0].text(
					-0.15,
					0.5,
					strategy.upper() if strategy != 'pretrained' else f"{strategy.capitalize()} {pretrained_model_arch}",
					transform=axes[row_idx][0].transAxes,
					fontsize=14,
					fontweight='bold',
					va='center',
					ha='right',
					rotation=90,
					color=border_color,
			)
	
	# Save the visualization
	file_name = os.path.join(
			results_dir,
			f"{dataset_name}_"
			f"Top{effective_topk}_images_"
			f"{img_hash}_"
			f"Q_{re.sub(' ', '_', query_text)}_"
			f"{'_'.join(all_strategies)}_"
			f"{re.sub(r'[/@]', '-', pretrained_model_arch)}_"
			f"t2i_merged.png"
	)
	
	plt.tight_layout()
	plt.savefig(file_name, bbox_inches='tight', dpi=dpi)
	plt.close()
	print(f"Saved visualization to: {file_name}")

def plot_text_to_images(
		models, 
		validation_loader, 
		preprocess, 
		query_text, 
		topk, 
		device, 
		results_dir, 
		cache_dir=None, 
		embeddings_cache=None, 
		dpi=200,
		scale_factor=10.0,
	):

	dataset_name = getattr(validation_loader, 'name', 'unknown_dataset')
	img_hash = hashlib.sha256(query_text.encode()).hexdigest()[:8]
	if cache_dir is None:
		cache_dir = results_dir
	
	tokenized_query = clip.tokenize([query_text]).to(device)
	
	for strategy, model in models.items():
		print(f"Processing strategy: {strategy} ".center(160, " "))
		if strategy == 'pretrained':
			model_arch = re.sub(r'[/@]', '-', model.name)
			print(f"{model.__class__.__name__} {model_arch}".center(160, " "))
		model.eval()
		print(f"[Text-to-image(s)] strategy: {strategy} Query: '{query_text}'".center(160, " "))
		
		# Get top-k images
		all_image_embeddings, image_paths = embeddings_cache[strategy]
		all_image_embeddings = all_image_embeddings.to(device, dtype=torch.float32)
		
		with torch.no_grad():
			text_features = model.encode_text(tokenized_query).to(torch.float32)
			text_features = F.normalize(text_features, dim=-1)
			similarities = (100.0 * text_features @ all_image_embeddings.T).softmax(dim=-1)
			effective_topk = min(topk, len(all_image_embeddings))
			topk_scores, topk_indices = torch.topk(similarities.squeeze(), effective_topk)
			topk_scores = topk_scores.cpu().numpy()
			topk_indices = topk_indices.cpu().numpy()
		
		# Get ground truth labels - handle both single and multi-label
		dataset = validation_loader.dataset
		try:
			ground_truth_labels = dataset.labels
			topk_ground_truth_labels = []
			
			for idx in topk_indices:
				gt_label = ground_truth_labels[idx]
				
				# Convert to list format for consistent handling
				if isinstance(gt_label, (list, tuple)):
					# Already multi-label
					label_list = [str(label).capitalize() for label in gt_label]
				elif isinstance(gt_label, str):
					# Check if it's a string representation of a list
					if gt_label.startswith('[') and gt_label.endswith(']'):
						# Parse string representation of list
						import ast
						try:
							parsed_list = ast.literal_eval(gt_label)
							if isinstance(parsed_list, (list, tuple)):
								label_list = [str(label).capitalize() for label in parsed_list]
							else:
								label_list = [str(parsed_list).capitalize()]
						except (ValueError, SyntaxError):
							# If parsing fails, treat as single label
							label_list = [gt_label.capitalize()]
					else:
						# Single label - convert to list
						label_list = [gt_label.capitalize()]
				else:
					# Handle tensor or other formats
					if hasattr(gt_label, 'tolist'):
						gt_list = gt_label.tolist()
						if isinstance(gt_list, list):
							label_list = [str(label).capitalize() for label in gt_list]
						else:
							label_list = [str(gt_list).capitalize()]
					else:
						label_list = [str(gt_label).capitalize()]
				
				topk_ground_truth_labels.append(label_list)
				
		except (AttributeError, IndexError) as e:
			print(f"Warning: Could not retrieve ground-truth labels: {e}")
			topk_ground_truth_labels = [[f"Unknown GT {idx}"] for idx in topk_indices]
		
		# Load all images first (same as original)
		topk_images = []
		for idx in topk_indices:
			try:
				img_path = image_paths[idx]
				if os.path.exists(img_path):
					img = Image.open(img_path).convert('RGB')
				else:
					sample = dataset[idx]
					if len(sample) >= 3:
						img = sample[0]
					else:
						raise ValueError(f"Unexpected dataset structure at index {idx}")
					
					if isinstance(img, torch.Tensor):
						img = img.cpu().numpy()
						if img.shape[0] in [1, 3]:
							img = img.transpose(1, 2, 0)
						mean = np.array([0.5754663102194626, 0.564594860510725, 0.5443646108296668])
						std = np.array([0.2736517370426002, 0.26753170455186887, 0.2619102890668636])
						img = img * std + mean
						img = np.clip(img, 0, 1)
						img = (img * 255).astype(np.uint8)
						img = Image.fromarray(img)
				topk_images.append(img)
			except Exception as e:
				print(f"Warning: Could not load image {idx}: {e}")
				blank_img = np.ones((224, 224, 3), dtype=np.uint8) * 128
				topk_images.append(Image.fromarray(blank_img))
		
		# Title height in pixels - increased to accommodate 3 lines of GT labels
		title_height = int(110 * scale_factor)
		
		# First determine dimensions
		heights = [img.height for img in topk_images]
		widths = [img.width for img in topk_images]
		
		# Use the same aspect ratio for all images
		max_height = max(heights)
		
		# Scale max_height to make images larger while preserving aspect ratio
		scaled_max_height = int(max_height * scale_factor)
		
		# Resize images to have same height and apply scaling
		for i in range(len(topk_images)):
			target_height = scaled_max_height
			target_width = int(topk_images[i].width * (target_height / topk_images[i].height))
			topk_images[i] = topk_images[i].resize((target_width, target_height), Image.LANCZOS)
		
		# Update widths after resizing
		widths = [img.width for img in topk_images]
		
		# Create a composite image
		total_width = sum(widths)
		print(f"Composite dimensions: {total_width} x {scaled_max_height + title_height}")
		
		composite = Image.new(
			mode='RGB',
			size=(total_width, scaled_max_height + title_height),
			color='white',
		)
		
		# Add each image
		x_offset = 0
		for i, img in enumerate(topk_images):
			composite.paste(img, (x_offset, title_height))  # Leave space at top for text
			x_offset += img.width
		
		# Scale font sizes based on scale_factor
		default_font_size_title = int(28 * scale_factor)
		default_font_size_score = int(20 * scale_factor)
		default_font_size_gt = int(12 * scale_factor)
		
		try:
			title_font = ImageFont.truetype("DejaVuSansMono-Bold.ttf", default_font_size_title)
			score_font = ImageFont.truetype("DejaVuSansMono.ttf", default_font_size_score)
			gt_font = ImageFont.truetype("NimbusSans-Regular.otf", default_font_size_gt)
		except IOError:
			try:
				title_font = ImageFont.truetype("NimbusSans-Bold.otf", default_font_size_title)
				score_font = ImageFont.truetype("NimbusSans-Regular.otf", default_font_size_score)
				gt_font = ImageFont.truetype("NimbusSans-Regular.otf", default_font_size_gt)
			except IOError:
				print("Warning: Could not load any fonts. Falling back to default font.")
				try:
					# Try this approach for PIL 9.0.0+
					title_font = ImageFont.load_default().font_variant(size=default_font_size_title)
					score_font = ImageFont.load_default().font_variant(size=default_font_size_score)
					gt_font = ImageFont.load_default().font_variant(size=default_font_size_gt)
				except AttributeError:
					# Fallback for older PIL versions
					title_font = score_font = gt_font = ImageFont.load_default()
		
		draw = ImageDraw.Draw(composite)
		
		# Add a subtle dividing line between title area and image
		draw.line(
			[(0, title_height-2), (total_width, title_height-2)], 
			fill="#DDDDDD", 
			width=int(1 * scale_factor)
		)
		
		# Add text for each image with proper vertical alignment
		x_offset = 0
		for i, (score, gt_labels, img) in enumerate(zip(topk_scores, topk_ground_truth_labels, topk_images)):
			# Calculate center position for this image section
			center_x = x_offset + img.width // 2
			
			# Prepare the score text
			score_text = f"Score: {score:.3f}"
			
			# Get text dimensions using appropriate method for the PIL version
			if hasattr(score_font, 'getbbox'):
				score_bbox = score_font.getbbox(score_text)
				score_width = score_bbox[2] - score_bbox[0]
				score_height_px = score_bbox[3] - score_bbox[1]
			else:
				score_width, score_height_px = score_font.getsize(score_text)
			
			# Draw "Score: X.X" text centered at the top
			score_y = int(5 * scale_factor)
			draw.text(
				(center_x - score_width//2, score_y),
				score_text,
				fill="black",
				font=score_font
			)
			
			# Draw GT labels - each on a separate line
			gt_start_y = score_y + score_height_px + int(10 * scale_factor)
			line_height = int(15 * scale_factor)  # Space between lines
			
			for j, gt_label in enumerate(gt_labels):
				# Get dimensions for this GT label
				if hasattr(gt_font, 'getbbox'):
					gt_bbox = gt_font.getbbox(gt_label)
					gt_width = gt_bbox[2] - gt_bbox[0]
				else:
					gt_width, _ = gt_font.getsize(gt_label)
				
				# Calculate y position for this line
				gt_y = gt_start_y + j * line_height
				
				# Center the text horizontally
				gt_x = center_x - gt_width // 2
				
				# Draw the GT label
				draw.text(
					(gt_x, gt_y),
					gt_label,
					fill="#0004EC",  # Different color to distinguish from score
					font=gt_font
				)
			
			x_offset += img.width
		
		# Save the composite image
		file_name = os.path.join(
			results_dir,
			f'{dataset_name}_'
			f'Top{effective_topk}_'
			f'images_{img_hash}_'
			f'Q_{re.sub(" ", "_", query_text)}_'
			f'{strategy}_'
			f'{model_arch}_'
			f't2i.png'
		)                
		composite.save(file_name, dpi=(dpi, dpi))
		print(f"Saved composite image to: {file_name}")

def plot_comparison_metrics_split_table_annotation(
				dataset_name: str,
				pretrained_img2txt_dict: dict,
				pretrained_txt2img_dict: dict,
				finetuned_img2txt_dict: dict,
				finetuned_txt2img_dict: dict,
				model_name: str,
				finetune_strategies: list,
				results_dir: str,
				topK_values: list,
				figure_size=(7, 6),
				DPI: int = 200,
		):
		metrics = ["mP", "mAP", "Recall"]
		modes = ["Image-to-Text", "Text-to-Image"]
		all_model_architectures = [
				'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64',
				'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px',
		]
		
		# Validate model exists in dictionaries
		if model_name not in finetuned_img2txt_dict.keys():
				print(f"WARNING: {model_name} not found in finetuned_img2txt_dict. Skipping...")
				print(json.dumps(finetuned_img2txt_dict, indent=4, ensure_ascii=False))
				return
		if model_name not in finetuned_txt2img_dict.keys():
				print(f"WARNING: {model_name} not found in finetuned_txt2img_dict. Skipping...")
				print(json.dumps(finetuned_txt2img_dict, indent=4, ensure_ascii=False))
				return
		
		# Validate finetune_strategies
		finetune_strategies = [s for s in finetune_strategies if s in ["full", "lora", "progressive"]][:3]  # Max 3
		if not finetune_strategies:
				print("WARNING: No valid finetune strategies provided. Skipping...")
				return
		
		model_name_idx = all_model_architectures.index(model_name) if model_name in all_model_architectures else 0
		
		# Define colors and styles for the different strategies
		strategy_colors = {'full': '#0058a5', 'lora': '#f58320be', 'progressive': '#cc40df'}  # Blue, Orange, Purple
		pretrained_colors = {'ViT-B/32': '#745555', 'ViT-B/16': '#9467bd', 'ViT-L/14': '#e377c2', 'ViT-L/14@336px': '#696969'}
		strategy_styles = {'full': 's', 'lora': '^', 'progressive': 'd'}  # Unique markers
		
		# Key K points for table annotations
		key_k_values = [1, 10, 20]
		
		# Process each mode (Image-to-Text and Text-to-Image)
		for mode in modes:
				pretrained_dict = pretrained_img2txt_dict if mode == "Image-to-Text" else pretrained_txt2img_dict
				finetuned_dict = finetuned_img2txt_dict if mode == "Image-to-Text" else finetuned_txt2img_dict
				
				# Process each metric (mP, mAP, Recall)
				for metric in metrics:
						# Create figure with adjusted size
						fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
						
						# Create filename for output
						fname = f"{dataset_name}_{'_'.join(finetune_strategies)}_finetune_vs_pretrained_CLIP_{re.sub(r'[/@]', '-', model_name)}_{mode.replace('-', '_')}_{metric}_comparison.png"
						file_path = os.path.join(results_dir, fname)
						
						# Check if metric exists in pretrained dictionary
						if metric not in pretrained_dict.get(model_name, {}):
								print(f"WARNING: Metric {metric} not found in pretrained_{mode.lower().replace('-', '_')}_dict for {model_name}")
								continue
								
						# Get available k values across all dictionaries
						k_values = sorted(
								k for k in topK_values if str(k) in pretrained_dict.get(model_name, {}).get(metric, {})
						)
						
						# Validate k values across all strategies
						for strategy in finetune_strategies:
								if strategy not in finetuned_dict.get(model_name, {}) or metric not in finetuned_dict.get(model_name, {}).get(strategy, {}):
										print(f"WARNING: Metric {metric} not found in finetuned_{mode.lower().replace('-', '_')}_dict for {model_name}/{strategy}")
										k_values = []  # Reset if any strategy is missing
										break
								k_values = sorted(
										set(k_values) & set(int(k) for k in finetuned_dict.get(model_name, {}).get(strategy, {}).get(metric, {}).keys())
								)
								
						if not k_values:
								print(f"WARNING: No matching K values found for {metric}")
								continue
						
						# Store lines for legend
						lines = []
								
						# Plot Pre-trained (dashed line)
						pretrained_vals = [pretrained_dict[model_name][metric].get(str(k), float('nan')) for k in k_values]
						pretrained_line, = ax.plot(
								k_values,
								pretrained_vals,
								label=f"CLIP {model_name}",
								color=pretrained_colors[model_name],
								linestyle='--', 
								marker='o',
								linewidth=1.5,
								markersize=4,
								alpha=0.75,
						)
						lines.append(pretrained_line)
						
						# Plot each Fine-tuned strategy (solid lines, thicker, distinct markers)
						strategy_lines = {}
						for strategy in finetune_strategies:
								finetuned_vals = [finetuned_dict[model_name][strategy][metric].get(str(k), float('nan')) for k in k_values]
								line, = ax.plot(
										k_values,
										finetuned_vals,
										label=f"{strategy.upper()}",
										color=strategy_colors[strategy], 
										linestyle='-', 
										marker=strategy_styles[strategy],
										linewidth=2.0, 
										markersize=5,
								)
								lines.append(line)
								strategy_lines[strategy] = finetuned_vals
						
						# Prepare data for table annotations at key K points
						tables_data = {}
						for k in key_k_values:
								if k in k_values:
										k_idx = k_values.index(k)
										pre_val = pretrained_vals[k_idx]
										
										# Collect improvements for all strategies at this K
										improvements = {}
										for strategy in finetune_strategies:
												if k_idx < len(strategy_lines[strategy]):
														ft_val = strategy_lines[strategy][k_idx]
														if pre_val != 0:
																imp_pct = (ft_val - pre_val) / pre_val * 100
																improvements[strategy] = (imp_pct, ft_val)
										
										# Sort strategies by improvement (descending)
										sorted_strategies = sorted(improvements.items(), key=lambda x: x[1][0], reverse=True)
										
										# Store for later use
										tables_data[k] = {
												'improvements': improvements,
												'sorted_strategies': sorted_strategies,
												'best_strategy': sorted_strategies[0][0] if sorted_strategies else None,
												'worst_strategy': sorted_strategies[-1][0] if sorted_strategies else None,
												'best_val': sorted_strategies[0][1][1] if sorted_strategies else None,
												'worst_val': sorted_strategies[-1][1][1] if sorted_strategies else None,
										}
						
						# Add table annotations for each key K point
						for k, data in tables_data.items():
								if not data['sorted_strategies']:
										continue
								
								# Create table text
								table_text = f"K={k}:\n"
								
								ranking_labels = ["1)", "2)", "3)"][:len(data['sorted_strategies'])]
								if len(data['sorted_strategies']) == 2:
										ranking_labels = ["1)", "2)"]  # Only 2 strategies
								
								for (strategy, (imp, _)), rank in zip(data['sorted_strategies'], ranking_labels):
										color_hex = strategy_colors[strategy]
										table_text += f"{rank} {strategy.upper()}: {imp:+.1f}%\n"  # Add each line to the table text
									
								if k == min(k_values):  # First K point (e.g., K=1)
										# Check if there's more space above best or below worst
										best_val = data['best_val']
										worst_val = data['worst_val']
										
										# Calculate available space
										space_above = 1.0 - best_val  # Space to top of plot
										space_below = worst_val - 0.0  # Space to bottom of plot
										
										# Position based on available space
										if space_above >= 0.2 or space_above > space_below:
												# Place above the highest point
												xy = (k, best_val)
												xytext = (10, 20)  # Offset to upper right
												va = 'bottom'
										else:
												# Place below the lowest point
												xy = (k, worst_val)
												xytext = (10, -20)  # Offset to lower right
												va = 'top'
								
								elif k == max(k_values):  # Last K point (e.g., K=20)
										# Similar logic but offset to the left
										best_val = data['best_val']
										worst_val = data['worst_val']
										
										space_above = 1.0 - best_val
										space_below = worst_val - 0.0
										
										if space_above >= 0.2 or space_above > space_below:
												xy = (k, best_val)
												xytext = (-10, 20)  # Offset to upper left
												va = 'bottom'
										else:
												xy = (k, worst_val)
												xytext = (-10, -20)  # Offset to lower left
												va = 'top'
								
								else:  # Middle K points (e.g., K=10)
										# Try to position in middle of plot if possible
										mid_y = (data['best_val'] + data['worst_val']) / 2
										xy = (k, mid_y)
										xytext = (0, 30 if mid_y < 0.5 else -30)  # Above if in lower half, below if in upper half
										va = 'bottom' if mid_y < 0.5 else 'top'
								
								# Add the annotation table
								ax.annotate(
										table_text,
										xy=xy,
										xytext=xytext,
										textcoords='offset points',
										fontsize=8,
										verticalalignment=va,
										horizontalalignment='center',
										bbox=dict(
												boxstyle="round,pad=0.4",
												facecolor='white',
												edgecolor='gray',
												alpha=0.9
										),
										zorder=10  # Ensure annotation is above other elements
								)
						
						# Format the plot
						ax.set_title(
								f"{metric}@K", 
								fontsize=10, 
								fontweight='bold',
						)
						ax.set_xlabel("K", fontsize=10, fontweight='bold')
						ax.set_xticks(k_values)
						ax.grid(True, linestyle='--', alpha=0.75)
						ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
						ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1.0'], fontsize=10)
						ax.set_ylim(-0.01, 1.01)
						ax.tick_params(axis='both', labelsize=7)

						# Set spine edge color to solid black
						for spine in ax.spines.values():
								spine.set_color('black')
								spine.set_linewidth(0.7)
								
						plt.tight_layout()
						plt.savefig(file_path, dpi=DPI, bbox_inches='tight')
						plt.close(fig)

def plot_comparison_metrics_split(
		dataset_name: str,
		pretrained_img2txt_dict: dict,
		pretrained_txt2img_dict: dict,
		finetuned_img2txt_dict: dict,
		finetuned_txt2img_dict: dict,
		model_name: str,
		finetune_strategies: list,
		results_dir: str,
		topK_values: list,
		figure_size=(9, 8),
		DPI: int = 300,
	):
	metrics = ["mP", "mAP", "Recall"]
	modes = ["Image-to-Text", "Text-to-Image"]
	all_model_architectures = [
		'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64',
		'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px',
	]
	if model_name not in finetuned_img2txt_dict.keys():
		print(f"WARNING: {model_name} not found in finetuned_img2txt_dict. Skipping...")
		print(json.dumps(finetuned_img2txt_dict, indent=4, ensure_ascii=False))
		return
	if model_name not in finetuned_txt2img_dict.keys():
		print(f"WARNING: {model_name} not found in finetuned_txt2img_dict. Skipping...")
		print(json.dumps(finetuned_txt2img_dict, indent=4, ensure_ascii=False))
		return
	
	# Validate finetune_strategies
	finetune_strategies = [s for s in finetune_strategies if s in ["full", "lora", "progressive"]][:3]  # Max 3
	if not finetune_strategies:
		print("WARNING: No valid finetune strategies provided. Skipping...")
		return
	model_name_idx = all_model_architectures.index(model_name) if model_name in all_model_architectures else 0
	
	# Define a professional color palette for fine-tuned strategies
	strategy_colors = {'full': '#0058a5', 'lora': '#f58320be', 'progressive': '#cc40df'}  # Blue, Orange, Purple
	pretrained_colors = {'ViT-B/32': '#745555', 'ViT-B/16': '#9467bd', 'ViT-L/14': '#e377c2', 'ViT-L/14@336px': '#696969'}
	strategy_styles = {'full': 's', 'lora': '^', 'progressive': 'd'}  # Unique markers
	
	for mode in modes:
		pretrained_dict = pretrained_img2txt_dict if mode == "Image-to-Text" else pretrained_txt2img_dict
		finetuned_dict = finetuned_img2txt_dict if mode == "Image-to-Text" else finetuned_txt2img_dict
		for metric in metrics:
			# Create figure with slightly adjusted size for better annotation spacing
			fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
			
			# Create filename for the output
			fname = f"{dataset_name}_{'_'.join(finetune_strategies)}_finetune_vs_pretrained_CLIP_{re.sub(r'[/@]', '-', model_name)}_{mode.replace('-', '_')}_{metric}_comparison.png"
			file_path = os.path.join(results_dir, fname)
			
			# Check if metric exists in pretrained dictionary
			if metric not in pretrained_dict.get(model_name, {}):
				print(f"WARNING: Metric {metric} not found in pretrained_{mode.lower().replace('-', '_')}_dict for {model_name}")
				continue
					
			# Get available k values across all dictionaries
			k_values = sorted(k for k in topK_values if str(k) in pretrained_dict.get(model_name, {}).get(metric, {}))
			
			# Validate k values across all strategies
			for strategy in finetune_strategies:
				if strategy not in finetuned_dict.get(model_name, {}) or metric not in finetuned_dict.get(model_name, {}).get(strategy, {}):
					print(f"WARNING: Metric {metric} not found in finetuned_{mode.lower().replace('-', '_')}_dict for {model_name}/{strategy}")
					k_values = []  # Reset if any strategy is missing
					break
				k_values = sorted(set(k_values) & set(int(k) for k in finetuned_dict.get(model_name, {}).get(strategy, {}).get(metric, {}).keys()))
					
			if not k_values:
				print(f"WARNING: No matching K values found for {metric}")
				continue
					
			# Plot Pre-trained (dashed line)
			pretrained_vals = [pretrained_dict[model_name][metric].get(str(k), float('nan')) for k in k_values]
			ax.plot(
				k_values,
				pretrained_vals,
				label=f"CLIP {model_name}",
				color=pretrained_colors[model_name],
				linestyle='--', 
				marker='o',
				linewidth=2.0,
				markersize=4,
				alpha=0.75,
			)
			
			# Plot each Fine-tuned strategy (solid lines, thicker, distinct markers)
			for strategy in finetune_strategies:
				finetuned_vals = [finetuned_dict[model_name][strategy][metric].get(str(k), float('nan')) for k in k_values]
				ax.plot(
					k_values,
					finetuned_vals,
					label=f"{strategy.upper()}",
					color=strategy_colors[strategy], 
					linestyle='-', 
					marker=strategy_styles[strategy],
					linewidth=2.5,
					markersize=5,
				)
			
			# Analyze plot data to place annotations intelligently
			key_k_values = [1, 10, 20]  # These are the key points to annotate
			annotation_positions = {}    # To store planned annotation positions
			
			# First pass: gather data about values and improvements
			for k in key_k_values:
				if k in k_values:
					k_idx = k_values.index(k)
					pre_val = pretrained_vals[k_idx]
					finetuned_vals_at_k = {strategy: finetuned_dict[model_name][strategy][metric].get(str(k), float('nan')) for strategy in finetune_strategies}
					
					# Calculate improvements
					improvements = {}
					for strategy, val in finetuned_vals_at_k.items():
						if pre_val != 0:
							imp = (val - pre_val) / pre_val * 100
							improvements[strategy] = (imp, val)
					
					# Store data for this k
					annotation_positions[k] = {
						'best': max(improvements.items(), key=lambda x: x[1][0]),
						'worst': min(improvements.items(), key=lambda x: x[1][0]),
						'all_values': [v[1] for v in improvements.values()]
					}
			
			# Second pass: determine optimal annotation placement based on plot density
			for k, data in annotation_positions.items():
				best_strategy, (best_imp, best_val) = data['best']
				worst_strategy, (worst_imp, worst_val) = data['worst']
				
				# Find y positions of all lines at this k
				all_values = data['all_values']
				all_values.sort()  # Sort for easier gap analysis
				
				# For best annotation (typically placed above)
				best_text_color = '#016e2bff' if best_imp >= 0 else 'red'
				best_arrow_style = '<|-' if best_imp >= 0 else '-|>'
				
				# For worst annotation (typically placed below)
				worst_text_color = '#016e2bff' if worst_imp >= 0 else 'red'
				worst_arrow_style = '-|>' if worst_imp >= 0 else '<|-'
				
				# Calculate the overall range and spacing between values
				if len(all_values) > 1:  # More than one strategy
					value_range = max(all_values) - min(all_values)
					avg_gap = value_range / (len(all_values) - 1) if len(all_values) > 1 else 0.1
					
					# Check if annotation positioning needs adjustment
					if value_range < 0.15:  # Values are close together
						# Use more extreme offsets
						best_offset = (5, 20)  # Further right and higher up
						worst_offset = (5, -20)  # Further right and lower down
					else:
						# Regular offsets for well-separated values
						best_offset = (0, 20)
						worst_offset = (0, -20)
				else:
					# Default offsets when there's only one strategy
					best_offset = (0, 30)
					worst_offset = (0, -30)
				
				# Place best strategy annotation with adjusted position
				ax.annotate(
					f"{best_imp:+.1f}%",
					xy=(k, best_val),
					xytext=best_offset,
					textcoords='offset points',
					fontsize=12,
					fontweight='bold',
					color=best_text_color,
					bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.1),
					arrowprops=dict(
						arrowstyle=best_arrow_style,
						color=best_text_color,
						shrinkA=0,
						shrinkB=3, # 
						alpha=0.8,
					)
				)
				
				# Place worst strategy annotation with adjusted position
				# Only annotate worst if it's different from best (avoids duplication)
				if worst_strategy != best_strategy:
					ax.annotate(
						f"{worst_imp:+.1f}%",
						xy=(k, worst_val),
						xytext=worst_offset,
						textcoords='offset points',
						fontsize=12,
						fontweight='bold',
						color=worst_text_color,
						bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.1),
						arrowprops=dict(
							arrowstyle=worst_arrow_style,
							color=worst_text_color,
							shrinkA=0,
							shrinkB=3,
							alpha=0.8,
						)
					)
			
			# Format the plot
			y_offset = 1.05
			title_bottom_y = y_offset + 0.02  # Calculate position below title
			legend_gap = 0.0  # Fixed gap between title and legend
			legend_y_pos = title_bottom_y - legend_gap
			ax.set_title(
				f"{metric}@K", 
				fontsize=13, 
				fontweight='bold', 
				y=y_offset,
			)
			ax.set_xlabel("K", fontsize=11, fontweight='bold')
			ax.set_xticks(k_values)
			ax.set_xticklabels(k_values, fontsize=15)
			# ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
			# ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1.0'], fontsize=10)
			ax.grid(True, linestyle='--', alpha=0.4, color='black')
			y_max = max(max(pretrained_vals), max(finetuned_vals))
			y_min = min(min(pretrained_vals), min(finetuned_vals))
			padding = (y_max - y_min) * 0.2
			print(f"{metric}@K y_min: {y_min}, y_max: {y_max} padding: {padding}")
			# ax.set_ylim(min(0, y_min - padding), max(1, y_max + padding))
			ax.set_ylim(max(-0.02, y_min - padding), min(1.02, y_max + padding))
			# ax.set_ylim(-0.01, 1.01)
			ax.set_yticklabels([f'{y:.2f}' for y in ax.get_yticks()], fontsize=15)
			
			ax.legend(
				loc='upper center',
				bbox_to_anchor=(0.5, legend_y_pos),  # Position with fixed gap below title
				frameon=False,
				fontsize=12,
				facecolor='white',
				ncol=len(finetune_strategies) + 1,
			)
			
			# Set spine edge color to solid black
			for spine in ax.spines.values():
				spine.set_color('black')
				spine.set_linewidth(0.7)
					
			plt.tight_layout()
			plt.savefig(file_path, dpi=DPI, bbox_inches='tight')
			plt.close(fig)

def plot_comparison_metrics_merged(
		dataset_name: str,
		pretrained_img2txt_dict: dict,
		pretrained_txt2img_dict: dict,
		finetuned_img2txt_dict: dict,
		finetuned_txt2img_dict: dict,
		model_name: str,
		finetune_strategies: list,
		results_dir: str,
		topK_values: list,
		figure_size=(15, 6),
		DPI: int = 300,
	):
		metrics = ["mP", "mAP", "Recall"]
		modes = ["Image-to-Text", "Text-to-Image"]
		all_model_architectures = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
		# Validate model_name and finetune_strategies
		finetune_strategies = [s for s in finetune_strategies if s in ["full", "lora", "progressive"]][:3]  # Max 3
		if not finetune_strategies:
				print("WARNING: No valid finetune strategies provided. Skipping...")
				return
		for mode in modes:
				finetuned_dict = finetuned_img2txt_dict if mode == "Image-to-Text" else finetuned_txt2img_dict
				if model_name not in finetuned_dict or not all(strategy in finetuned_dict.get(model_name, {}) for strategy in finetune_strategies):
						print(f"WARNING: Some strategies for {model_name} not found in finetuned_{mode.lower().replace('-', '_')}_dict. Skipping...")
						return
		model_name_idx = all_model_architectures.index(model_name) if model_name in all_model_architectures else 0
		strategy_colors = {'full': '#0058a5', 'lora': '#f58320be', 'progressive': '#cc40df'}  # Blue, Orange, Green
		pretrained_colors = {'ViT-B/32': '#745555', 'ViT-B/16': '#9467bd', 'ViT-L/14': '#e377c2', 'ViT-L/14@336px': '#696969'}
		strategy_styles = {'full': 's', 'lora': '^', 'progressive': 'd'}  # Unique markers

		for i, mode in enumerate(modes):
				fig, axes = plt.subplots(1, 3, figsize=figure_size, constrained_layout=True)
				fname = f"{dataset_name}_{'_'.join(finetune_strategies)}_finetune_vs_pretrained_CLIP_{re.sub(r'[/@]', '-', model_name)}_retrieval_performance_comparison_{mode.replace('-', '_')}_merged.png"
				file_path = os.path.join(results_dir, fname)
				fig.suptitle(
						f'$\\it{{{mode}}}$ Retrieval Performance Comparison\n'
						f'Pre-trained CLIP {model_name} vs. {", ".join(s.capitalize() for s in finetune_strategies)} Fine-tuning',
						fontsize=12, fontweight='bold',
				)
				pretrained_dict = pretrained_img2txt_dict if mode == "Image-to-Text" else pretrained_txt2img_dict
				finetuned_dict = finetuned_img2txt_dict if mode == "Image-to-Text" else finetuned_txt2img_dict
				for j, metric in enumerate(metrics):
						ax = axes[j]
						k_values = sorted(
								k for k in topK_values if str(k) in pretrained_dict.get(model_name, {}).get(metric, {})
						)
						for strategy in finetune_strategies:
								k_values = sorted(
										set(k_values) & set(int(k) for k in finetuned_dict.get(model_name, {}).get(strategy, {}).get(metric, {}).keys())
								)
						if not k_values:
								print(f"WARNING: No matching K values for {mode}, {metric}")
								continue
						# Plot Pre-trained (dashed line)
						pretrained_values = [pretrained_dict.get(model_name, {}).get(metric, {}).get(str(k), float('nan')) for k in k_values]
						ax.plot(
								k_values, pretrained_values,
								label=f"Pre-trained CLIP {model_name}",
								color=pretrained_colors[model_name], marker='o', linestyle='--',
								linewidth=1.5, markersize=5, alpha=0.7,
						)
						# Plot each Fine-tuned strategy (solid lines, thicker, distinct markers)
						for strategy in finetune_strategies:
								finetuned_values = [finetuned_dict.get(model_name, {}).get(strategy, {}).get(metric, {}).get(str(k), float('nan')) for k in k_values]
								ax.plot(
										k_values, finetuned_values,
										label=f"{strategy.capitalize()} Fine-tune",
										color=strategy_colors[strategy], marker=strategy_styles[strategy], linestyle='-',
										linewidth=2.5, markersize=6,
								)
						# Find the best and worst performing strategies at key K values
						key_k_values = [1, 10, 20]
						for k in key_k_values:
								if k in k_values:
										k_idx = k_values.index(k)
										pre_val = pretrained_values[k_idx]
										finetuned_vals_at_k = {strategy: finetuned_dict.get(model_name, {}).get(strategy, {}).get(metric, {}).get(str(k), float('nan')) for strategy in finetune_strategies}
										# Find best and worst strategies
										best_strategy = max(finetuned_vals_at_k, key=finetuned_vals_at_k.get)
										worst_strategy = min(finetuned_vals_at_k, key=finetuned_vals_at_k.get)
										best_val = finetuned_vals_at_k[best_strategy]
										worst_val = finetuned_vals_at_k[worst_strategy]
										# Annotate best strategy (green)
										if pre_val != 0:
												best_imp = (best_val - pre_val) / pre_val * 100
												# Set color based on improvement value
												text_color = '#016e2bff' if best_imp >= 0 else 'red'
												arrow_style = '<|-' if best_imp >= 0 else '-|>'
												
												# Place annotations with arrows
												ax.annotate(
													f"{best_imp:+.1f}%",
													xy=(k, best_val),
													xytext=(0, 30),
													textcoords='offset points',
													fontsize=8,
													fontweight='bold',
													color=text_color,
													bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.3),
													arrowprops=dict(
														arrowstyle=arrow_style,
														color=text_color,
														shrinkA=0,
														shrinkB=3,
														alpha=0.8,
														connectionstyle="arc3,rad=.2"
													)
												)
										# Annotate worst strategy (red)
										if pre_val != 0:
												worst_imp = (worst_val - pre_val) / pre_val * 100
												# Set color based on improvement value
												text_color = 'red' if worst_imp <= 0 else '#016e2bff'
												arrow_style = '-|>' if worst_imp >= 0 else '<|-'
												
												# Place annotations with arrows
												ax.annotate(
													f"{worst_imp:+.1f}%",
													xy=(k, worst_val),
													xytext=(0, -30),
													textcoords='offset points',
													fontsize=8,
													fontweight='bold',
													color=text_color,
													bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.3),
													arrowprops=dict(
														arrowstyle=arrow_style,
														color=text_color,
														shrinkA=0,
														shrinkB=3,
														alpha=0.8,
														connectionstyle="arc3,rad=.2"
													)
												)
						# Axes formatting
						ax.set_xlabel('K', fontsize=11)
						ax.set_title(f'{metric}@K', fontsize=10, fontweight='bold')
						ax.grid(True, linestyle='--', alpha=0.9)
						ax.set_xticks(k_values)
						ax.set_ylim(-0.01, 1.01)
						if j == 0:
								ax.legend(fontsize=9, loc='best')
						# Set spine edge color to solid black
						for spine in ax.spines.values():
								spine.set_color('black')
								spine.set_linewidth(1.0)
				plt.savefig(file_path, dpi=DPI, bbox_inches='tight')
				plt.close(fig)

		# Overall summary
		print(f"\n{'='*80}")
		print(f"OVERALL PERFORMANCE SUMMARY [QUANTITATIVE ANALYSIS]")
		print(f"{'='*80}")
		for mode in modes:
				pretrained_dict = pretrained_img2txt_dict if mode == "Image-to-Text" else pretrained_txt2img_dict
				finetuned_dict = finetuned_img2txt_dict if mode == "Image-to-Text" else finetuned_txt2img_dict
				print(f"\nMode: {mode}")
				for metric in metrics:
						k_values = sorted(
								k for k in topK_values if str(k) in pretrained_dict.get(model_name, {}).get(metric, {})
						)
						for strategy in finetune_strategies:
								k_values = sorted(
										set(k_values) & set(int(k) for k in finetuned_dict.get(model_name, {}).get(strategy, {}).get(metric, {}).keys())
								)
						if k_values:
								for strategy in finetune_strategies:
										improvements = [
												((finetuned_dict[model_name][strategy][metric][str(k)] - pretrained_dict[model_name][metric][str(k)]) /
												 pretrained_dict[model_name][metric][str(k)] * 100)
												for k in k_values if pretrained_dict[model_name][metric][str(k)] != 0
										]
										if improvements:
												avg_improvement = sum(improvements) / len(improvements)
												print(f"  {metric} ({strategy.capitalize()}): Average improvement across all K values: {avg_improvement:+.2f}%")
		print(f"\n{'='*80}")

def plot_all_pretrain_metrics(
		dataset_name: str,
		img2txt_metrics_dict: dict,
		txt2img_metrics_dict: dict,
		topK_values: list,
		results_dir: str,
		figure_size=(12, 5),
		DPI: int=300,
	):
	metrics = ["mP", "mAP", "Recall"]
	modes = ["Image-to-Text", "Text-to-Image"]
	models = list(img2txt_metrics_dict.keys()) # ['RN50', 'RN101', ..., 'ViT-L/14@336px']
	
	# Use distinct colors, markers, and linestyles for each model
	colors = plt.cm.Set1.colors
	markers = ['D', 'v', 'o', 's', '^', 'p', 'h', '*', 'H'] # 9 distinct markers
	linestyles = [':', '-.', '-', '--', '-', '--', ':', '-.', '-'] # Cycle through styles
	
	# Create separate plots for each mode (Image-to-Text and Text-to-Image)
	for i, mode in enumerate(modes):
		# Determine which metrics dictionary to use based on the mode
		metrics_dict = img2txt_metrics_dict if mode == "Image-to-Text" else txt2img_metrics_dict
		
		# Create a filename for each plot
		file_name = f"{dataset_name}_{len(models)}_pretrained_clip_models_{mode.replace('-', '_').lower()}_{'_'.join(re.sub(r'[/@]', '-', m) for m in models)}.png"
		
		# Create a figure with 1x3 subplots (one for each metric)
		fig, axes = plt.subplots(1, len(metrics), figsize=figure_size, constrained_layout=True)
		# fig.suptitle(f"{dataset_name} Pre-trained CLIP - {mode} Retrieval Metrics", fontsize=11, fontweight='bold')
		fig.suptitle(f"Pre-trained CLIP {mode} Retrieval Metrics", fontsize=11, fontweight='bold')
		
		# Create a plot for each metric
		for j, metric in enumerate(metrics):
			ax = axes[j]
			legend_handles = []
			legend_labels = []
			
			# Plot data for each model
			for k, (model_name, color, marker, linestyle) in enumerate(zip(models, colors, markers, linestyles)):
				if model_name in metrics_dict:
					k_values = sorted([int(k) for k in metrics_dict[model_name][metric].keys() if int(k) in topK_values])
					values = [metrics_dict[model_name][metric][str(k)] for k in k_values]
					
					line, = ax.plot(
						k_values,
						values,
						label=model_name,
						color=color,
						marker=marker,
						linestyle=linestyle,
						linewidth=1.8,
						markersize=5,
					)
					
					legend_handles.append(line)
					legend_labels.append(model_name)
				
				# Configure the axes and labels
				ax.set_xlabel('K', fontsize=10)
				ax.set_ylabel(f'{metric}@K', fontsize=10)
				ax.set_title(f'{metric}@K', fontsize=12, fontweight="bold")
				ax.grid(True, linestyle='--', alpha=0.9)
				ax.set_xticks(topK_values)
				ax.set_xlim(min(topK_values) - 1, max(topK_values) + 1)
				
				# Set dynamic y-axis limits
				all_values = [v for m in models if m in metrics_dict for v in [metrics_dict[m][metric][str(k)] for k in k_values]]
				if all_values:
					min_val = min(all_values)
					max_val = max(all_values)
					padding = 0.05 * (max_val - min_val) if (max_val - min_val) > 0 else 0.05
					ax.set_ylim(bottom=0, top=max(1.01, max_val + padding))
			
		# Add legend at the bottom of the figure
		fig.legend(
			legend_handles,
			legend_labels,
			title="Image Encoder",
			title_fontsize=10,
			fontsize=9,
			loc='lower center',
			ncol=min(len(models), 5),  # Limit to 5 columns for readability
			bbox_to_anchor=(0.5, 0.01),
			bbox_transform=fig.transFigure,
			frameon=True,
			shadow=True,
			fancybox=True,
			edgecolor='black',
			facecolor='white',
		)
		
		# Adjust layout and save figure
		plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # Make room for the legend at bottom
		plt.savefig(os.path.join(results_dir, file_name), dpi=DPI, bbox_inches='tight')
		plt.close(fig)

def visualize_samples(dataloader, dataset, num_samples=5):
		for bidx, (images, tokenized_labels, labels_indices) in enumerate(dataloader):
				print(f"Batch {bidx}, Shapes: {images.shape}, {tokenized_labels.shape}, {labels_indices.shape}")
				if bidx >= num_samples:
						break
				
				# Get the global index of the first sample in this batch
				start_idx = bidx * dataloader.batch_size
				for i in range(min(dataloader.batch_size, len(images))):
						global_idx = start_idx + i
						if global_idx >= len(dataset):
								break
						image = images[i].permute(1, 2, 0).numpy()  # Convert tensor to numpy array
						caption_idx = labels_indices[i].item()
						path = dataset.images[global_idx]
						label = dataset.labels[global_idx]
						label_int = dataset.labels_int[global_idx]
						
						print(f"Global Index: {global_idx}")
						print(f"Image {image.shape} Path: {path}")
						print(f"Label: {label}, Label Int: {label_int}, Caption Index: {caption_idx}")
						
						# Denormalize the image (adjust mean/std based on your dataset)
						mean = np.array([0.5126933455467224, 0.5045100450515747, 0.48094621300697327])
						std = np.array([0.276103675365448, 0.2733437418937683, 0.27065524458885193])
						image = image * std + mean  # Reverse normalization
						image = np.clip(image, 0, 1)  # Ensure pixel values are in [0, 1]
						
						plt.figure(figsize=(10, 10))
						plt.imshow(image)
						plt.title(f"Label: {label} (Index: {caption_idx})")
						plt.axis('off')
						plt.show()

def visualize_(dataloader, batches=3, num_samples=5):
	"""
	Visualize the first 'num_samples' images of each batch in a single figure.
	Args:
			dataloader (torch.utils.data.DataLoader): Data loader containing images and captions.
			num_samples (int, optional): Number of batches to visualize. Defaults to 5.
			num_cols (int, optional): Number of columns in the visualization. Defaults to 5.
	"""
	# Get the number of batches in the dataloader
	num_batches = len(dataloader)
	# Limit the number of batches to visualize
	num_batches = min(num_batches, batches)
	# Create a figure with 'num_samples' rows and 'num_cols' columns
	fig, axes = plt.subplots(nrows=num_batches, ncols=num_samples, figsize=(20, num_batches * 2))
	# Iterate over the batches
	for bidx, (images, tokenized_labels, labels_indices) in enumerate(dataloader):
		if bidx >= num_batches:
			break
		# Iterate over the first 'num_cols' images in the batch
		for cidx in range(num_samples):
			image = images[cidx].permute(1, 2, 0).numpy()  # Convert tensor to numpy array and permute dimensions
			caption_idx = labels_indices[cidx]
			# Denormalize the image
			image = image * np.array([0.2268645167350769]) + np.array([0.6929051876068115])
			image = np.clip(image, 0, 1)  # Ensure pixel values are in [0, 1] range
			# Plot the image
			axes[bidx, cidx].imshow(image)
			axes[bidx, cidx].set_title(f"Batch {bidx+1}, Img {cidx+1}: {caption_idx}", fontsize=8)
			axes[bidx, cidx].axis('off')
	# Layout so plots do not overlap
	plt.tight_layout()
	plt.show()

def plot_retrieval_metrics_best_model(
		dataset_name: str,
		image_to_text_metrics: Dict[str, Dict[str, float]],
		text_to_image_metrics: Dict[str, Dict[str, float]],
		fname: str ="Retrieval_Performance_Metrics_best_model.png",
		best_model_name: str ="Best Model",
		figure_size=(11, 4),
		DPI: int=300,
	):
	metrics = list(image_to_text_metrics.keys())  # ['mP', 'mAP', 'Recall']
	suptitle_text = f"{dataset_name} Retrieval Performance Metrics [{best_model_name}]: "
	for metric in metrics:
		suptitle_text += f"{metric}@K | " 
	suptitle_text = suptitle_text[:-3]  # Remove trailing " | "
	modes = ['Image-to-Text', 'Text-to-Image']
	
	fig, axes = plt.subplots(1, len(metrics), figsize=figure_size, constrained_layout=True)
	fig.suptitle(suptitle_text, fontsize=8, fontweight='bold')
	
	# Store legend handles and labels
	legend_handles = []
	legend_labels = []

	for i, metric in enumerate(metrics):
		ax = axes[i] if len(metrics) > 1 else axes # Handle single subplot case
		# print(f"Image-to-Text:")
		it_top_ks = list(map(int, image_to_text_metrics[metric].keys()))  # K values for Image-to-Text
		it_vals = list(image_to_text_metrics[metric].values())
		# print(metric, it_top_ks, it_vals)
		line, = ax.plot(
			it_top_ks, 
			it_vals, 
			marker='o',
			label=modes[0], 
			color='blue',
			linestyle='-',
			linewidth=1.0,
			markersize=2.0,
		)
		if modes[0] not in legend_labels:
			legend_handles.append(line)
			legend_labels.append(modes[0])
		
		# Plotting for Text-to-Image
		# print(f"Text-to-Image:")
		ti_top_ks = list(map(int, text_to_image_metrics[metric].keys()))  # K values for Text-to-Image
		ti_vals = list(text_to_image_metrics[metric].values())
		# print(metric, ti_top_ks, ti_vals)
		line, = ax.plot(
			ti_top_ks,
			ti_vals,
			marker='s',
			label=modes[1],
			color='red',
			linestyle='-',
			linewidth=1.0,
			markersize=2.0,
		)
		if modes[1] not in legend_labels:
			legend_handles.append(line)
			legend_labels.append(modes[1])
		
		ax.set_xlabel('K', fontsize=8)
		ax.set_ylabel(f'{metric}@K', fontsize=8)
		ax.set_title(f'{metric}@K', fontsize=9, fontweight="bold")
		ax.grid(True, linestyle='--', alpha=0.7)
		
		# Set the x-axis to only show integer values
		all_ks = sorted(set(it_top_ks + ti_top_ks))
		ax.set_xticks(all_ks)

		# Adjust y-axis to start from 0 for better visualization
		# ax.set_ylim(bottom=-0.05, top=1.05)
		all_values = it_vals + ti_vals
		min_val = min(all_values)
		max_val = max(all_values)
		padding = 0.02 * (max_val - min_val) if (max_val - min_val) > 0 else 0.02
		ax.set_ylim(bottom=min(-0.02, min_val - padding), top=max(0.5, max_val + padding))

	plt.tight_layout(rect=[0, 0.03, 1, 0.95])
	fig.legend(
		legend_handles,
		legend_labels,
		fontsize=9,
		loc='upper center',
		ncol=len(modes),
		bbox_to_anchor=(0.5, 0.94),
		bbox_transform=fig.transFigure,
		frameon=True,
		shadow=True,
		fancybox=True,
		edgecolor='black',
		facecolor='white',
	)
	plt.savefig(fname, dpi=DPI, bbox_inches='tight')
	plt.close(fig)

def plot_retrieval_metrics_per_epoch(
		dataset_name: str,
		image_to_text_metrics_list: List[Dict[str, Dict[str, float]]],
		text_to_image_metrics_list: List[Dict[str, Dict[str, float]]],
		fname: str = "Retrieval_Performance_Metrics.png",
	):
	num_epochs = len(image_to_text_metrics_list)
	num_xticks = min(10, num_epochs)
	epochs = range(1, num_epochs + 1)
	selective_xticks_epochs = np.linspace(0, num_epochs, num_xticks, dtype=int)
	if num_epochs < 2:
		return
	# Derive valid K values from the metrics for each mode
	if image_to_text_metrics_list and text_to_image_metrics_list:
		# Get K values from the first epoch's metrics for each mode
		it_first_metrics = image_to_text_metrics_list[0]["mP"]  # Use "mP" as a representative metric
		ti_first_metrics = text_to_image_metrics_list[0]["mP"]  # Use "mP" as a representative metric
		it_valid_k_values = sorted([int(k) for k in it_first_metrics.keys()])  # K values for Image-to-Text
		ti_valid_k_values = sorted([int(k) for k in ti_first_metrics.keys()])  # K values for Text-to-Image
		# Print warning if K values differ significantly (optional, for debugging)
		if set(it_valid_k_values) != set(ti_valid_k_values):
			print(f"<!> Warning: K values differ between Image-to-Text ({it_valid_k_values}) and Text-to-Image ({ti_valid_k_values}).")

	modes = ["Image-to-Text", "Text-to-Image"]
	metrics = list(image_to_text_metrics_list[0].keys())  # ['mP', 'mAP', 'Recall']
	
	suptitle_text = f"{dataset_name} Retrieval Performance Metrics [per epoch]: "
	for metric in metrics:
		suptitle_text += f"{metric}@K | "
	suptitle_text = suptitle_text[:-3]  # Remove trailing " | "
	
	markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'H', 'x']  # Larger, distinct markers for each line
	line_styles = ['-', '--', ':', '-.', '-']  # Varied line styles for clarity
	# colors = plt.cm.tab10.colors  # Use a color map for distinct colors
	colors = plt.cm.Set1.colors
	fig, axs = plt.subplots(len(modes), len(metrics), figsize=(20, 11), constrained_layout=True)
	fig.suptitle(suptitle_text, fontsize=15, fontweight='bold')
	# Store legend handles and labels
	legend_handles = []
	legend_labels = []
	for i, task_metrics_list in enumerate([image_to_text_metrics_list, text_to_image_metrics_list]):
		for j, metric in enumerate(metrics):
			ax = axs[i, j]
			# Use the appropriate K values for each mode
			valid_k_values = it_valid_k_values if i == 0 else ti_valid_k_values
			all_values = []
			for k_idx, (K, color, marker, linestyle) in enumerate(zip(valid_k_values, colors, markers, line_styles)):
				values = []
				for metrics_dict in task_metrics_list:
					if metric in metrics_dict and str(K) in metrics_dict[metric]:
						values.append(metrics_dict[metric][str(K)])
					else:
						values.append(0)  # Default to 0 if K value is missing (shouldn’t happen with valid data)
				all_values.extend(values)
				line, = ax.plot(
					epochs,
					values,
					label=f'K={K}',
					color=color,
					alpha=0.9,
					linewidth=1.8,
				)
				if f'K={K}' not in legend_labels:
					legend_handles.append(line)
					legend_labels.append(f'K={K}')

			ax.set_xlabel('Epoch', fontsize=12)
			ax.set_ylabel(f'{metric}@K', fontsize=12)
			ax.set_title(f'{modes[i]} - {metric}@K', fontsize=14)
			ax.grid(True, linestyle='--', alpha=0.7)
			# ax.set_xticks(epochs)
			ax.set_xticks(selective_xticks_epochs) # Only show selected epochs
			ax.set_xlim(0, num_epochs + 1)
			# ax.set_ylim(bottom=-0.05, top=1.05)
			# Dynamic y-axis limits
			if all_values:
				min_val = min(all_values)
				max_val = max(all_values)
				padding = 0.02 * (max_val - min_val) if (max_val - min_val) > 0 else 0.02
				# ax.set_ylim(bottom=min(-0.05, min_val - padding), top=max(1.05, max_val + padding))
				ax.set_ylim(bottom=-0.01, top=min(1.05, max_val + padding))
			else:
				ax.set_ylim(bottom=-0.01, top=1.05)
	
	# fig.legend(
	# 	legend_handles,
	# 	legend_labels,
	# 	fontsize=11,
	# 	loc='upper center',
	# 	ncol=len(legend_labels),  # Adjust number of columns based on number of K values
	# 	bbox_to_anchor=(0.5, 0.96),
	# 	bbox_transform=fig.transFigure,
	# 	frameon=True,
	# 	shadow=True,
	# 	fancybox=True,
	# 	edgecolor='black',
	# 	facecolor='white',
	# )
	# plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
	plt.savefig(fname, dpi=300, bbox_inches='tight')
	plt.close(fig)

def plot_loss_accuracy_metrics(
    dataset_name: str,
    train_losses: List[float],
    val_losses: List[float],
    in_batch_topk_val_accuracy_i2t_list: Optional[List[Dict[int, float]]] = None,
    in_batch_topk_val_accuracy_t2i_list: Optional[List[Dict[int, float]]] = None,
    full_topk_val_accuracy_i2t_list: Optional[List[Dict[int, float]]] = None,
    full_topk_val_accuracy_t2i_list: Optional[List[Dict[int, float]]] = None,
    mean_reciprocal_rank_list: Optional[List[float]] = None,
    cosine_similarity_list: Optional[List[float]] = None,
    losses_file_path: str = "losses.png",
    in_batch_topk_val_acc_i2t_fpth: str = "in_batch_val_topk_accuracy_i2t.png",
    in_batch_topk_val_acc_t2i_fpth: str = "in_batch_val_topk_accuracy_t2i.png",
    full_topk_val_acc_i2t_fpth: str = "full_val_topk_accuracy_i2t.png",
    full_topk_val_acc_t2i_fpth: str = "full_val_topk_accuracy_t2i.png",
    mean_reciprocal_rank_file_path: str = "mean_reciprocal_rank.png",
    cosine_similarity_file_path: str = "cosine_similarity.png",
    DPI: int = 300,
    figure_size: Tuple[int, int] = (10, 4),
) -> None:
    """
    Plot training/validation loss curves and a variety of top‑K accuracy / ranking metrics.

    The function is defensive:
        * If a metric list is empty or its first element does not contain any keys,
          the corresponding plot is silently skipped.
        * Legend `ncol` is forced to be at least 1, so ``len(topk_values)==0`` no longer
          triggers a ValueError.
    """
    num_epochs = len(train_losses)
    if num_epochs <= 1:
        # Nothing worth plotting
        return

    # -----------------------------------------------------------------
    # Common helpers
    # -----------------------------------------------------------------
    epochs = np.arange(1, num_epochs + 1)

    # For readability we show at most 20 x‑ticks
    num_xticks = min(20, num_epochs)
    selective_xticks = np.linspace(1, num_epochs, num_xticks, dtype=int)

    colors = {
        "train": "#1f77b4",
        "val": "#ff7f0e",
        "img2txt": "#2ca02c",
        "txt2img": "#d62728",
    }

    def setup_plot(ax, xlabel="Epoch", ylabel=None, title=None):
        ax.set_xlabel(xlabel, fontsize=12)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12)
        if title:
            ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlim(0, num_epochs + 1)
        ax.set_xticks(selective_xticks)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, linestyle="--", alpha=0.7)
        return ax

    # -----------------------------------------------------------------
    # 1️⃣  Loss curve
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figure_size)
    ax.plot(
        epochs,
        train_losses,
        color=colors["train"],
        label="Training",
        lw=1.5,
        marker="o",
        markersize=2,
    )
    ax.plot(
        epochs,
        val_losses,
        color=colors["val"],
        label="Validation",
        lw=1.5,
        marker="o",
        markersize=2,
    )
    setup_plot(ax, ylabel="Loss", title=f"{dataset_name} Learning Curve (Loss)")
    ax.legend(
        fontsize=10,
        loc="best",
        frameon=True,
        fancybox=True,
        shadow=True,
        facecolor="white",
        edgecolor="black",
    )
    fig.tight_layout()
    fig.savefig(losses_file_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # -----------------------------------------------------------------
    # Helper to plot any Top‑K metric (in‑batch or full)
    # -----------------------------------------------------------------
    def _plot_topk(
        metric_list: List[Dict[int, float]],
        fpath: str,
        direction: str,
        match_type: str,
    ):
        """
        Parameters
        ----------
        metric_list : list of dict
            One dict per epoch, mapping K → accuracy.
        fpath : str
            Where to save the figure.
        direction : {"i2t", "t2i"}
            Image‑to‑Text or Text‑to‑Image.
        match_type : {"in‑batch", "full"}
            Kind of retrieval evaluation.
        """
        if not metric_list:
            return

        # Defensive: make sure the first element actually has keys
        first_elem = metric_list[0]
        if not isinstance(first_elem, dict) or len(first_elem) == 0:
            return

        topk_values = sorted(first_elem.keys())
        fig, ax = plt.subplots(figsize=figure_size)

        for i, k in enumerate(topk_values):
            acc_vals = [epoch_dict.get(k, np.nan) for epoch_dict in metric_list]
            ax.plot(
                epochs,
                acc_vals,
                label=f"Top-{k}",
                lw=1.5,
                marker="o",
                markersize=2,
                color=plt.cm.tab10(i % 10),
            )

        title = f"{dataset_name} {direction.upper()} Top‑K [{match_type}] Validation Accuracy"
        setup_plot(ax, ylabel="Accuracy", title=title)
        ax.set_ylim(-0.05, 1.05)

        # ``ncol`` must be >= 1 – protect against empty ``topk_values``
        ncol = max(1, len(topk_values))
        ax.legend(
            fontsize=9,
            loc="best",
            ncol=ncol,
            frameon=True,
            fancybox=True,
            shadow=True,
            facecolor="white",
            edgecolor="black",
        )
        fig.tight_layout()
        fig.savefig(fpath, dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    # -----------------------------------------------------------------
    # 2️⃣  Image‑to‑Text top‑K plots
    # -----------------------------------------------------------------
    _plot_topk(
        in_batch_topk_val_accuracy_i2t_list,
        in_batch_topk_val_acc_i2t_fpth,
        direction="i2t",
        match_type="in‑batch",
    )
    _plot_topk(
        full_topk_val_accuracy_i2t_list,
        full_topk_val_acc_i2t_fpth,
        direction="i2t",
        match_type="full",
    )

    # -----------------------------------------------------------------
    # 3️⃣  Text‑to‑Image top‑K plots
    # -----------------------------------------------------------------
    _plot_topk(
        in_batch_topk_val_accuracy_t2i_list,
        in_batch_topk_val_acc_t2i_fpth,
        direction="t2i",
        match_type="in‑batch",
    )
    _plot_topk(
        full_topk_val_accuracy_t2i_list,
        full_topk_val_acc_t2i_fpth,
        direction="t2i",
        match_type="full",
    )

    # -----------------------------------------------------------------
    # 4️⃣  Mean Reciprocal Rank (optional)
    # -----------------------------------------------------------------
    if mean_reciprocal_rank_list:
        fig, ax = plt.subplots(figsize=figure_size)
        ax.plot(
            epochs,
            mean_reciprocal_rank_list,
            color="#9467bd",
            label="MRR",
            lw=1.5,
            marker="o",
            markersize=2,
        )
        setup_plot(
            ax,
            ylabel="Mean Reciprocal Rank",
            title=f"{dataset_name} Mean Reciprocal Rank (Image‑to‑Text)",
        )
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=10, loc="best", frameon=True)
        fig.tight_layout()
        fig.savefig(mean_reciprocal_rank_file_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    # -----------------------------------------------------------------
    # 5️⃣  Cosine Similarity (optional)
    # -----------------------------------------------------------------
    if cosine_similarity_list:
        fig, ax = plt.subplots(figsize=figure_size)
        ax.plot(
            epochs,
            cosine_similarity_list,
            color="#17becf",
            label="Cosine Similarity",
            lw=1.5,
            marker="o",
            markersize=2,
        )
        setup_plot(
            ax,
            ylabel="Cosine Similarity",
            title=f"{dataset_name} Cosine Similarity Between Embeddings",
        )
        ax.legend(fontsize=10, loc="best")
        fig.tight_layout()
        fig.savefig(cosine_similarity_file_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)

def plot_loss_accuracy_metrics_old(
		dataset_name: str,
		train_losses: List[float],
		val_losses: List[float],
		in_batch_topk_val_accuracy_i2t_list: List[float],
		in_batch_topk_val_accuracy_t2i_list: List[float],
		full_topk_val_accuracy_i2t_list: List[float] = None,
		full_topk_val_accuracy_t2i_list: List[float] = None,
		mean_reciprocal_rank_list: List[float] = None,
		cosine_similarity_list: List[float] = None,
		losses_file_path: str = "losses.png",
		in_batch_topk_val_acc_i2t_fpth: str = "in_batch_val_topk_accuracy_i2t.png",
		in_batch_topk_val_acc_t2i_fpth: str = "in_batch_val_topk_accuracy_t2i.png",
		full_topk_val_acc_i2t_fpth: str = "full_val_topk_accuracy_i2t.png",
		full_topk_val_acc_t2i_fpth: str = "full_val_topk_accuracy_t2i.png",
		mean_reciprocal_rank_file_path: str = "mean_reciprocal_rank.png",
		cosine_similarity_file_path: str = "cosine_similarity.png",
		DPI: int = 300,
		figure_size: tuple = (10, 4),
	):

	num_epochs = len(train_losses)
	if num_epochs <= 1:  # No plot if only one epoch
		return
			
	# Setup common plotting configurations
	epochs = np.arange(1, num_epochs + 1)
	
	# Create selective x-ticks for better readability
	num_xticks = min(20, num_epochs)
	selective_xticks = np.linspace(1, num_epochs, num_xticks, dtype=int)
	
	# Define a consistent color palette
	colors = {
		'train': '#1f77b4',
		'val': '#ff7f0e',
		'img2txt': '#2ca02c',
		'txt2img': '#d62728'
	}
	
	# Common plot settings function
	def setup_plot(ax, xlabel='Epoch', ylabel=None, title=None):
		ax.set_xlabel(xlabel, fontsize=12)
		if ylabel:
			ax.set_ylabel(ylabel, fontsize=12)
		if title:
			ax.set_title(title, fontsize=10, fontweight='bold')
		ax.set_xlim(0, num_epochs + 1)
		ax.set_xticks(selective_xticks)
		ax.tick_params(axis='both', labelsize=10)
		ax.grid(True, linestyle='--', alpha=0.7)
		return ax
	
	# 1. Losses plot
	fig, ax = plt.subplots(figsize=figure_size)
	ax.plot(
		epochs,
		train_losses, 
		color=colors['train'], 
		label='Training', 
		lw=1.5, 
		marker='o', 
		markersize=2,
	)
	ax.plot(
		epochs,
		val_losses,
		color=colors['val'], 
		label='Validation',
		lw=1.5, 
		marker='o', 
		markersize=2,
	)
					
	setup_plot(
		ax, ylabel='Loss', 
		title=f'{dataset_name} Learning Curve: (Loss)',
	)
	ax.legend(
		fontsize=10, 
		loc='best', 
		frameon=True, 
		fancybox=True,
		shadow=True,
		facecolor='white',
		edgecolor='black',
	)
	fig.tight_layout()
	fig.savefig(losses_file_path, dpi=DPI, bbox_inches='tight')
	plt.close(fig)
	
	# 1. Image-to-Text Top-K[in-batch matching] Validation Accuracy plot
	if in_batch_topk_val_accuracy_i2t_list:
		topk_values = list(in_batch_topk_val_accuracy_i2t_list[0].keys())
		fig, ax = plt.subplots(figsize=figure_size)
		
		for i, k in enumerate(topk_values):
			accuracy_values = [epoch_data[k] for epoch_data in in_batch_topk_val_accuracy_i2t_list]
			ax.plot(
				epochs, 
				accuracy_values, 
				label=f'Top-{k}',
				lw=1.5, 
				marker='o', 
				markersize=2, 
				color=plt.cm.tab10(i),
			)
							 
		setup_plot(
			ax, 
			ylabel='Accuracy', 
			title=f'{dataset_name} Image-to-Text Top-K [in-batch matching] Validation Accuracy'
		)
		ax.set_ylim(-0.05, 1.05)
		ax.legend(
			fontsize=9, 
			loc='best',
			ncol=len(topk_values),
			frameon=True, 
			fancybox=True,
			shadow=True,
			facecolor='white',
			edgecolor='black',
		)
		fig.tight_layout()
		fig.savefig(in_batch_topk_val_acc_i2t_fpth, dpi=DPI, bbox_inches='tight')
		plt.close(fig)

	# 2. Image-to-Text Top-K[full matching] Validation Accuracy plot
	if full_topk_val_accuracy_i2t_list:
		topk_values = list(full_topk_val_accuracy_i2t_list[0].keys())
		fig, ax = plt.subplots(figsize=figure_size)

		for i, k in enumerate(topk_values):
			accuracy_values = [epoch_data[k] for epoch_data in full_topk_val_accuracy_i2t_list]
			ax.plot(
				epochs,
				accuracy_values,
				label=f'Top-{k}',
				lw=1.5,
				marker='o',
				markersize=2,
				color=plt.cm.tab10(i),
			)

		setup_plot(
			ax,
			ylabel='Accuracy',
			title=f'{dataset_name} Image-to-Text Top-K [full matching] Validation Accuracy'
		)
		ax.set_ylim(-0.05, 1.05)
		ax.legend(
			fontsize=9,
			loc='best',
			ncol=len(topk_values),
			frameon=True,
			fancybox=True,
			shadow=True,
			edgecolor='black',
			facecolor='white',
		)
		fig.tight_layout()
		fig.savefig(full_topk_val_acc_i2t_fpth, dpi=DPI, bbox_inches='tight')
		plt.close(fig)

	if full_topk_val_accuracy_t2i_list:
		topk_values = list(full_topk_val_accuracy_t2i_list[0].keys())
		fig, ax = plt.subplots(figsize=figure_size)
		for i, k in enumerate(topk_values):
			accuracy_values = [epoch_data[k] for epoch_data in full_topk_val_accuracy_t2i_list]
			ax.plot(
				epochs,
				accuracy_values,
				label=f'Top-{k}',
				lw=1.5,
				marker='o',
				markersize=2,
				color=plt.cm.tab10(i),
			)

		setup_plot(
			ax,
			ylabel='Accuracy',
			title=f'{dataset_name} Text-to-Image Top-K [full matching] Validation Accuracy'
		)
		ax.set_ylim(-0.05, 1.05)
		ax.legend(
			fontsize=9,
			loc='best',
			ncol=len(topk_values),
			frameon=True,
			fancybox=True,
			shadow=True,
			edgecolor='black',
			facecolor='white',
		)
		fig.tight_layout()
		fig.savefig(full_topk_val_acc_t2i_fpth, dpi=DPI, bbox_inches='tight')
		plt.close(fig)

	# 3. Text-to-Image Top-K Accuracy plot
	if in_batch_topk_val_accuracy_t2i_list:
		topk_values = list(in_batch_topk_val_accuracy_t2i_list[0].keys())
		fig, ax = plt.subplots(figsize=figure_size)
		
		for i, k in enumerate(topk_values):
			accuracy_values = [epoch_data[k] for epoch_data in in_batch_topk_val_accuracy_t2i_list]
			ax.plot(
				epochs, 
				accuracy_values, 
				label=f'Top-{k}',
				lw=1.5, 
				marker='o', 
				markersize=2, 
				color=plt.cm.tab10(i),
			)
							 
		setup_plot(
			ax, 
			ylabel='Accuracy', 
			title=f'{dataset_name} Text-to-Image Top-K [in-batch matching] Validation Accuracy'
		)
		ax.set_ylim(-0.05, 1.05)
		ax.legend(
			fontsize=9, 
			loc='best',
			ncol=len(topk_values),
			frameon=True, 
			fancybox=True, 
			shadow=True,
			edgecolor='black',
			facecolor='white',
		)
		fig.tight_layout()
		fig.savefig(in_batch_topk_val_acc_t2i_fpth, dpi=DPI, bbox_inches='tight')
		plt.close(fig)
	
	# 4. Mean Reciprocal Rank plot (if data provided)
	if mean_reciprocal_rank_list and len(mean_reciprocal_rank_list) > 0:
		fig, ax = plt.subplots(figsize=figure_size)
		ax.plot(
			epochs, 
			mean_reciprocal_rank_list, 
			color='#9467bd', 
			label='MRR', 
			lw=1.5, 
			marker='o', 
			markersize=2,
		)
						
		setup_plot(
			ax, 
			ylabel='Mean Reciprocal Rank',
			title=f'{dataset_name} Mean Reciprocal Rank (Image-to-Text)',
		)
		
		ax.set_ylim(-0.05, 1.05)
		ax.legend(fontsize=10, loc='best', frameon=True)
		fig.tight_layout()
		fig.savefig(mean_reciprocal_rank_file_path, dpi=DPI, bbox_inches='tight')
		plt.close(fig)
	
	# 5. Cosine Similarity plot (if data provided)
	if cosine_similarity_list and len(cosine_similarity_list) > 0:
		fig, ax = plt.subplots(figsize=figure_size)
		ax.plot(
			epochs, 
			cosine_similarity_list, 
			color='#17becf',
			label='Cosine Similarity', 
			lw=1.5, 
			marker='o', 
			markersize=2,
		)
						
		setup_plot(
			ax, 
			ylabel='Cosine Similarity',
			title=f'{dataset_name} Cosine Similarity Between Embeddings'
		)
		ax.legend(fontsize=10, loc='best')
		fig.tight_layout()
		fig.savefig(cosine_similarity_file_path, dpi=DPI, bbox_inches='tight')
		plt.close(fig)
