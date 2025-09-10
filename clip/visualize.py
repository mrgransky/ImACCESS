from utils import *

strategy_colors = {
	'full': "#012894", 
	'lora': '#f58320be', 
	'progressive': '#cc40df',
	'probe': "#00C468D3",
}
strategy_styles = {
	'full': 's', 
	'lora': '^', 
	'progressive': 'd',
	'probe': 'x',
}
pretrained_colors = {
	'ViT-B/32': "#725151", 
	'ViT-B/16': '#9467bd', 
	'ViT-L/14': '#e377c2', 
	'ViT-L/14@336px': '#696969'
}

positive_pct_col = "#357402ff"
negative_pct_col = "#c0003aff"

transition_color = "#00A336"
early_stop_color = "#1D0808"
best_model_color = "#C002A7"
train_loss_color = "#0010F3"
val_loss_color = "#C27E00"
loss_imp_color = "#004214"
trainable_param_color = "#0104C9"

if USER == "farid":
	from graphviz import Digraph

def calculate_advanced_phase_metrics(phase_epochs, val_losses, phase):
	if not phase_epochs or len(phase_epochs) < 2:
		return {}
	
	# Get phase loss values
	phase_losses = [val_losses[e-1] for e in phase_epochs if 0 <= e-1 < len(val_losses)]
	if len(phase_losses) < 2:
		return {}
	
	# 1. Robust improvement (using moving averages to reduce noise)
	window = min(3, len(phase_losses) // 3)
	if window >= 1:
		start_avg = np.mean(phase_losses[:window])
		end_avg = np.mean(phase_losses[-window:])
		robust_improvement = ((start_avg - end_avg) / start_avg * 100) if start_avg > 0 else 0
	else:
		robust_improvement = 0
	
	# 2. Learning efficiency (improvement per epoch)
	duration = len(phase_losses)
	efficiency = robust_improvement / duration if duration > 0 else 0
	
	# 3. Convergence quality (how consistent was the improvement?)
	if len(phase_losses) > 2:
		epochs_array = np.arange(len(phase_losses))
		slope, intercept, r_value, _, _ = scipy.stats.linregress(epochs_array, phase_losses)
		convergence_quality = r_value ** 2  # R² indicates trend consistency
		learning_rate_metric = -slope  # Negative slope means improvement
	else:
		convergence_quality = 0
		learning_rate_metric = 0
	
	# 4. Volatility (coefficient of variation for normalized comparison)
	mean_loss = np.mean(phase_losses)
	volatility = np.std(phase_losses) / mean_loss if mean_loss > 0 else 0
	
	# 5. Early vs late learning
	mid_point = len(phase_losses) // 2
	if mid_point > 0:
		first_half_avg = np.mean(phase_losses[:mid_point])
		second_half_avg = np.mean(phase_losses[mid_point:])
		early_vs_late = ((first_half_avg - second_half_avg) / first_half_avg * 100) if first_half_avg > 0 else 0
	else:
		early_vs_late = 0
	
	# 6. Stability score (inverse of coefficient of variation)
	stability = (1 / volatility) if volatility > 0 else 0
	
	return {
		'robust_improvement': robust_improvement,
		'efficiency': efficiency,
		'convergence_quality': convergence_quality,
		'volatility': volatility,
		'learning_rate': learning_rate_metric,
		'early_vs_late': early_vs_late,
		'stability': stability,
		'duration': duration,
		'mean_loss': mean_loss,
		'final_loss': phase_losses[-1],
		'best_loss': min(phase_losses)
	}

def _short_label(mod: torch.nn.Module, max_len: int = 120) -> str:
		"""
		Return a concise, GraphViz‑safe label for *mod*.

		• Shows the class name.
		• For Conv2d / Linear / MultiheadAttention we also show the most
			important dimensions (e.g. 3→768, k=32×32).
		• Truncates anything longer than *max_len* characters.
		• Escapes double‑quotes and turns real new‑lines into the literal “\\n”.
		"""
		cls = mod.__class__.__name__

		if isinstance(mod, torch.nn.Conv2d):
				k = "×".join(map(str, mod.kernel_size))
				s = "×".join(map(str, mod.stride))
				txt = f"{cls}({mod.in_channels}→{mod.out_channels}, k={k}, s={s})"
		elif isinstance(mod, torch.nn.Linear):
				txt = f"{cls}({mod.in_features}→{mod.out_features})"
		elif isinstance(mod, torch.nn.MultiheadAttention):
				txt = f"{cls}(embed={mod.embed_dim}, heads={mod.num_heads})"
		elif isinstance(mod, torch.nn.LayerNorm):
				txt = f"{cls}({mod.normalized_shape[0]})"
		elif isinstance(mod, torch.nn.Embedding):
				txt = f"{cls}({mod.num_embeddings}→{mod.embedding_dim})"
		elif isinstance(mod, torch.nn.Dropout):
				txt = f"{cls}(p={mod.p})"
		else:
				# generic containers (Sequential, ModuleList, …)
				txt = cls

		# make it DOT‑safe
		txt = txt.replace('"', r'\"')          # escape "
		txt = txt.replace("\n", r"\n")        # literal "\n"
		if len(txt) > max_len:
				txt = txt[: max_len - 3] + "..."
		return txt

def _module_color(mod: torch.nn.Module) -> str:
		"""Pastel colour per layer type – helps the eye skim the graph."""
		if isinstance(mod, torch.nn.Conv2d):
				return "#AED6F1"   # light blue
		if isinstance(mod, torch.nn.Linear):
				return "#A9DFBF"   # light green
		if isinstance(mod, torch.nn.MultiheadAttention):
				return "#F9E79F"   # light yellow
		if isinstance(mod, torch.nn.LayerNorm):
				return "#F5CBA7"   # light orange
		if isinstance(mod, torch.nn.Embedding):
				return "#D7BDE2"   # light purple
		if isinstance(mod, torch.nn.Dropout):
				return "#E6B0AA"   # light red
		return "#D5DBDB"       # default – light grey for generic containers

def build_arch_flowchart(
		model: torch.nn.Module,
		*,
		filename: str = "clip_arch_flowchart",
		format: str = "png",
		view: bool = False,
		rankdir: str = "TB",                     # TB = top‑to‑bottom, LR = left‑to‑right
		canvas_inches: Tuple[float, float] = (30, 30),
		dpi: int = 250,
		ranksep: float = 1.8,
		nodesep: float = 1.8,
		node_fontsize: int = 12,
	):
		"""
		Create a GraphViz flow‑chart that mirrors the hierarchical structure
		of a ``torch.nn.Module``.
		"""
		# --------------------------------------------------------------
		# 1️⃣  Build a tree that mirrors the dotted module names
		# --------------------------------------------------------------
		tree: Dict[str, Any] = {}
		module_lookup: Dict[str, torch.nn.Module] = {}

		for name, mod in model.named_modules():
				parts = name.split(".") if name else []
				cur = tree
				for p in parts:
						cur = cur.setdefault(p, {})
				cur["__module__"] = mod
				module_lookup[name] = mod

		# --------------------------------------------------------------
		# 2️⃣  Initialise the Digraph (canvas size, fonts, etc.)
		# --------------------------------------------------------------
		graph = Digraph(
				name=filename,
				format=format,
				graph_attr={
						"rankdir": rankdir,
						"splines": "ortho",
						"bgcolor": "white",
						"size": f"{canvas_inches[0]},{canvas_inches[1]}",   # inches
						"dpi": str(dpi),
						"ranksep": str(ranksep),
						"nodesep": str(nodesep),
				},
				node_attr={
						"shape": "box",
						"style": "filled",
						"fontsize": str(node_fontsize),
						"fontname": "Helvetica",
				},
				edge_attr={"arrowhead": "none"},
		)

		# --------------------------------------------------------------
		# 3️⃣  Recursive helper that adds clusters and nodes
		# --------------------------------------------------------------
		def _add_subgraph(parent_name: str, subtree: Dict[str, Any], depth: int = 0):
				"""
				parent_name – dotted name of the current container
				subtree     – dict with children (and possibly "__module__")
				"""
				cluster_id = (
						f"cluster_{parent_name.replace('.', '_')}"
						if parent_name
						else "cluster_root"
				)
				with graph.subgraph(name=cluster_id) as c:
						# Title of the cluster (last component of the dotted name,
						# or the root class name for the top‑level cluster)
						title = parent_name.split(".")[-1] if parent_name else model.__class__.__name__
						c.attr(label=title, labelloc="t", fontsize="12", fontname="Helvetica-Bold")

						# ---- node for the *container itself* (only if it is NOT the root) ----
						# The root node is unnecessary – the cluster title already shows it.
						if parent_name and "__module__" in subtree:
								mod = subtree["__module__"]
								node_id = parent_name
								c.node(
										node_id,
										label=_short_label(mod),
										fillcolor=_module_color(mod),
										tooltip=str(mod),
								)

						# ---- walk over the children ---------------------------------------
						for child_name, child_subtree in subtree.items():
								if child_name == "__module__":
										continue
								full_name = f"{parent_name}.{child_name}" if parent_name else child_name

								# Leaf node – a real nn.Module
								if "__module__" in child_subtree:
										mod = child_subtree["__module__"]
										node_id = full_name
										graph.node(
												node_id,
												label=_short_label(mod),
												fillcolor=_module_color(mod),
												tooltip=str(mod),
										)
										# Edge from container (or from the root) to the leaf
										src = parent_name if parent_name else "root"
										graph.edge(src, node_id)

								# Recurse deeper – may be a sub‑container
								_add_subgraph(full_name, child_subtree, depth + 1)

		# --------------------------------------------------------------
		# 4️⃣  Build the whole chart
		# --------------------------------------------------------------
		_add_subgraph(parent_name="", subtree=tree)

		# --------------------------------------------------------------
		# 5️⃣  Render the file
		# --------------------------------------------------------------
		out_path = graph.render(filename=filename, cleanup=True, view=view)
		return graph

def plot_phase_transition_analysis_individual(
		training_history: Dict,
		file_path: str,
		figsize: Tuple[int, int] = (13, 7)
	):
	file_path = file_path.replace("_ph_anls.png", ".png")
	# Extract data
	epochs = [e + 1 for e in training_history['epochs']]  # 1-based indexing
	train_losses = training_history['train_losses']
	val_losses = training_history['val_losses']
	learning_rates = training_history['learning_rates']
	weight_decays = training_history['weight_decays']
	phases = training_history['phases']
	embedding_drifts = training_history['embedding_drifts']
	transitions = training_history.get('phase_transitions', [])
	early_stop_epoch = training_history.get('early_stop_epoch')
	best_epoch = training_history.get('best_epoch')
	
	# Color scheme
	phase_colors = plt.cm.tab10(np.linspace(0, 1, max(phases) + 1))

	# Helper to save figure with suffix
	def save_fig(fig, suffix):
		base, ext = os.path.splitext(file_path)
		fig.savefig(f"{base}_{suffix}{ext}", dpi=200, bbox_inches='tight', facecolor='white')
		plt.close(fig)
	
	# =============================================
	# PLOT 1: Learning Curve with Phase Transitions
	# =============================================
	fig, ax1 = plt.subplots(figsize=figsize, facecolor='white')
	max_loss = max(max(train_losses), max(val_losses))
	min_loss = min(min(train_losses), min(val_losses))
	margin = max_loss * 0.25
	ax1.set_ylim(min_loss - margin, max_loss + margin)
	ymin, ymax = ax1.get_ylim()
	y_middle = (ymin + ymax) / 2.0

	# Phase shading
	unique_phases = sorted(set(phases))
	for phase in set(phases):
		phase_epochs = [e for e, p in zip(epochs, phases) if p == phase]
		if phase_epochs:
			ax1.axvspan(min(phase_epochs), max(phase_epochs), alpha=0.39, color=phase_colors[phase], label=f'Phase {phase}')

	# Loss curves
	ax1.plot(epochs, train_losses, color="#0025FA", linewidth=2.5, alpha=0.9, label="Training Loss")
	ax1.plot(epochs, val_losses, color="#C77203", linewidth=2.5, alpha=0.9, label="Validation Loss")

	# Transitions
	for i, t_epoch in enumerate(transitions):
		ax1.axvline(
			x=t_epoch, 
			color=transition_color, 
			linestyle="--", 
			linewidth=1.5, 
			alpha=0.65,
			zorder=10,
		)
		if t_epoch < len(val_losses):
			prev_loss = val_losses[t_epoch - 1] if t_epoch > 0 else val_losses[t_epoch]
			change = ((prev_loss - val_losses[t_epoch]) / prev_loss) * 100 if prev_loss > 0 else 0
			ax1.text(
				t_epoch + 0.25,
				y_middle,
				f"T{i+1} ({change:+.2f}%)", 
				rotation=90,
				va="center", 
				ha="left", 
				color=transition_color, 
				fontsize=9,
			)

	if best_epoch is not None:
		ax1.scatter(
			[epochs[best_epoch]], 
			[val_losses[best_epoch]], 
			color=best_model_color, 
			marker="*", 
			s=150,
			linewidth=1.5, 
			zorder=15, 
			label="Best",
		)

	if early_stop_epoch:
		ax1.axvline(x=early_stop_epoch, color=early_stop_color, linestyle=":", linewidth=1.8, alpha=0.9)
		ax1.text(
			early_stop_epoch + 0.5, 
			y_middle, 
			"Early Stopping", 
			rotation=90, 
			va="center",
			ha="left", 
			color=early_stop_color, 
			fontsize=9
		)
	ax1.set_title("Learning Curve [Loss] with Phase Transitions", fontsize=10, weight="bold")
	ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
	ax1.legend(
		fontsize=8, 
		loc="best",
		ncol=len(unique_phases)+3,
		frameon=False, 
		shadow=False, 
		fancybox=False, 
		edgecolor="none", 
		facecolor="none",		
	)
	ax1.grid(True, alpha=0.5)
	save_fig(fig, "learning_curve")

	# ===================================
	# Hyperparameter Adaptation (LR + WD)
	# ===================================
	fig, ax2 = plt.subplots(figsize=figsize, facecolor='white')

	# Plot Learning Rate on primary axis
	for i in range(len(epochs) - 1):
		x0, x1 = epochs[i], epochs[i+1]
		y0, y1 = learning_rates[i], learning_rates[i+1]
		phase_color = phase_colors[phases[i+1]]
		ax2.plot(
			[x0, x1],
			[y0, y1],
			color=phase_color,
			linewidth=1.1,
			alpha=0.4,
			linestyle='-'
		)

	# Create twin axis for Weight Decay
	ax2_twin = ax2.twinx()

	# Plot Weight Decay on secondary axis
	for i in range(len(epochs) - 1):
		x0, x1 = epochs[i], epochs[i+1]
		y0, y1 = weight_decays[i], weight_decays[i+1]
		phase_color = phase_colors[phases[i+1]]
		ax2_twin.plot(
			[x0, x1],
			[y0, y1],
			color=phase_color,
			linewidth=2.0,
			alpha=0.5,
			linestyle='--'  # Different line style for distinction
		)

	for transition_epoch in transitions:
		if transition_epoch < len(learning_rates):
			ax2.axvline(
				x=transition_epoch,
				color=transition_color,
				linewidth=2.0,
				alpha=0.6,
				linestyle=':',
				zorder=10
			)

	ax2.set_xlabel('Epoch', fontsize=8, weight='bold')
	ax2.set_ylabel('LR', fontsize=7, weight='bold')
	ax2_twin.set_ylabel('WD', fontsize=7, weight='bold')
	ax2.tick_params(axis='y', labelsize=7)
	ax2_twin.tick_params(axis='y', labelsize=7)
	ax2.set_title('Hyperparameter Adaptation Across Phases\nLearning Rate (—) Weight Decay (--)', fontsize=8, weight='bold')
	ax2.grid(True, alpha=0.3, linestyle='-', color='#A3A3A3')
	ax2_twin.grid(True, alpha=0.5, linestyle='--', color="#8A8A8A")
	ax2.set_xlim(left=0, right=max(epochs)+1)
	ax2_twin.set_xlim(left=0, right=max(epochs)+1)
	save_fig(fig, "hyperparam_evol")

	# ===============================
	# Learning Rate Adaptation
	# ===============================
	fig, ax2 = plt.subplots(figsize=figsize, facecolor='white')
	for i in range(len(epochs) - 1):
		ax2.plot(
			[epochs[i], epochs[i+1]], 
			[learning_rates[i], learning_rates[i+1]],
			color=phase_colors[phases[i+1]], 
			linewidth=2.0,
		)
	for t_epoch in transitions:
		ax2.axvline(
			x=t_epoch, 
			color=transition_color, 
			linestyle=":",
			linewidth=2,
			alpha=0.8,
			zorder=10,
		)
	ax2.set_title("Learning Rate Adaptation Across Phases", fontsize=10, weight="bold")
	ax2.set_xlabel("Epoch")
	ax2.set_ylabel("LR", fontsize=8, weight="bold")
	ax2.grid(True, alpha=0.5, linestyle='--', color="#8A8A8A")
	save_fig(fig, "lr_evol")

	# ===============================
	# Weight Decay Adaptation
	# ===============================
	fig, ax3 = plt.subplots(figsize=figsize, facecolor='white')
	for i in range(len(epochs) - 1):
		ax3.plot(
			[epochs[i], epochs[i+1]], 
			[weight_decays[i], weight_decays[i+1]],
			color=phase_colors[phases[i+1]], 
			linewidth=2, 
			alpha=0.8
		)
	for t_epoch in transitions:
		ax3.axvline(
			x=t_epoch, 
			color=transition_color, 
			linestyle=":", 
			linewidth=2,
			alpha=0.8,
			zorder=10,
		)
	ax3.set_title("Weight Decay Adaptation Across Phases", fontsize=10, weight="bold")
	ax3.set_xlabel("Epoch")
	ax3.set_ylabel("WD", fontsize=8, weight="bold")
	ax3.grid(True, alpha=0.5, linestyle='--', color="#8A8A8A")
	save_fig(fig, "wd_evol")

	# ============================================
	# PLOT 4: Phase Efficiency Analysis
	# ============================================
	fig, ax4 = plt.subplots(figsize=figsize, facecolor='white')

	unique_phases = sorted(set(phases))
	phase_data = []
	for phase in unique_phases:
		phase_epochs = [e for e, p in zip(epochs, phases) if p == phase]
		duration = len(phase_epochs)
		if phase_epochs:
			s_idx, e_idx = phase_epochs[0] - 1, phase_epochs[-1] - 1 # 0-based indexing
			if 0 <= s_idx < len(val_losses) and 0 <= e_idx < len(val_losses):
				improvement = ((val_losses[s_idx] - val_losses[e_idx]) / val_losses[s_idx] * 100) if val_losses[s_idx] > 0 else 0
			else:
				improvement = 0
		else:
			improvement = 0
		phase_data.append((phase, duration, improvement))

	phases_list, durations, improvements = zip(*phase_data)

	bars = ax4.bar(
		range(len(durations)), 
		durations, 
		color=[phase_colors[p] for p in phases_list], 
		alpha=0.8
	)
	ax4_twin = ax4.twinx()
	ax4_twin.plot(
		range(len(improvements)),
		improvements,
		linestyle='-',
		marker='o',
		linewidth=1.0,
		markersize=2,
		color=loss_imp_color,
		)
	for i, (bar, imp) in enumerate(zip(bars, improvements)):
		ax4_twin.text(
			i,
			1.02*imp if imp > 0 else 0.9*imp,
			f"{imp:.2f}%",
			ha="center",
			fontsize=8,
			color=loss_imp_color,
			fontweight="bold",
		)

	ax4.set_title("Phase Efficiency Analysis", fontsize=9, weight="bold")
	ax4.set_xlabel("Phase", fontsize=7, weight="bold")
	ax4.set_ylabel("Epochs", color=trainable_param_color, fontsize=7, weight="bold")
	ax4.set_xticks(range(len(phases_list)))
	ax4.set_xticklabels([f"{p}" for p in phases_list])
	ax4.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
	ax4_twin.set_ylabel("Loss Improvement (%)", color=loss_imp_color, fontsize=7, weight="bold")
	ax4_twin.yaxis.set_tick_params(labelsize=8)

	# Match spine colors with their labels
	ax4.spines['left'].set_color(trainable_param_color)
	ax4_twin.spines['right'].set_color(loss_imp_color)

	# Hide the right and top spines for both axes
	ax4.grid(axis='y', alpha=0.5)
	ax4.tick_params(axis='y', labelcolor=trainable_param_color)
	ax4_twin.tick_params(axis='y', labelcolor=loss_imp_color)

	# Match spine colors with their labels
	ax4.spines['left'].set_color(trainable_param_color)
	ax4_twin.spines['right'].set_color(loss_imp_color)

	ax4_twin.spines['top'].set_visible(False)
	ax4.spines['top'].set_visible(False)

	save_fig(fig, "ph_eff")

	# ============================================
	# PLOT 5: Hyperparameter Correlations
	# ============================================
	fig, ax5 = plt.subplots(figsize=figsize, facecolor='white')
	lr_norm = np.array(learning_rates) / max(learning_rates)
	wd_norm = np.array(weight_decays) / max(weight_decays)
	loss_norm = np.array(val_losses) / max(val_losses)
	ax5.plot(epochs, lr_norm, "g-", label="LR")
	ax5.plot(epochs, wd_norm, "m-", label="WD")
	ax5.plot(epochs, loss_norm, "r-", label="Val Loss")
	for t_epoch in transitions:
		ax5.axvline(x=t_epoch, color=transition_color, linestyle=":", linewidth=1.5)
	ax5.set_title("Hyperparameter Correlations [normed]", fontsize=10, weight="bold")
	ax5.set_xlabel("Epoch"); ax5.set_ylabel("Normalized values")
	ax5.grid(True, alpha=0.3)
	ax5.legend(loc="best", fontsize=8, ncol=3, frameon=False)
	ax5.set_ylim(0, 1.1)
	save_fig(fig, "hp_corr")

	# ============================================
	# PLOT 6: Embedding Drift
	# ============================================
	fig, ax6 = plt.subplots(figsize=figsize, facecolor='white')
	ax6.set_title('Embedding Drift from Pre-trained State', fontsize=10, weight='bold')
	_seen_phases = set()
	for i in range(len(epochs) - 1):
		phase_index = phases[i]
		label = f'Phase {phase_index}' if phase_index not in _seen_phases else None
		_seen_phases.add(phase_index)
		ax6.plot(
			[epochs[i], epochs[i+1]], 
			[embedding_drifts[i], embedding_drifts[i+1]], 
			color=phase_colors[phase_index], 
			linewidth=2.5,
			marker='o', 
			markersize=3,
			alpha=0.9,
			label=label, # Show label only once per phase
		)

	_transition_label_shown = False
	for transition_epoch in transitions:
		label = 'Transition' if not _transition_label_shown else None
		_transition_label_shown = True
		ax6.axvline(
			x=transition_epoch,
			color=transition_color, 
			linestyle=':', 
			linewidth=1.5,
			alpha=0.5,
			label=label, # Show label only once
		)
	ax6.set_xlabel('Epoch', fontsize=10, fontweight="bold")
	ax6.set_ylabel('Drift (1 - CosSim)', fontsize=10, fontweight="bold")
	ax6.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.5)
	ax6.set_ylim(bottom=0) # Drift cannot be negative
	ax6.tick_params(axis='both', which='major', labelsize=8)
	ax6.legend(loc="best", fontsize=9, frameon=False)
	save_fig(fig, "emb_drift")

def plot_phase_transition_analysis(
		training_history: Dict,
		file_path: str,
		figsize: Tuple[int, int] = (19, 12)
	):
	# Extract data
	epochs = training_history['epochs']
	epochs = list(map(lambda x: x + 1, epochs)) # convert to 1-based indexing
	train_losses = training_history['train_losses']
	val_losses = training_history['val_losses']
	learning_rates = training_history['learning_rates']
	weight_decays = training_history['weight_decays']
	trainable_params_per_phase = training_history['trainable_params_per_phase']
	total_model_params = training_history['total_model_params']
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
	
	phase_colors = plt.cm.tab10(np.linspace(0, 1, max(phases) + 1))
	
	# ========================================
	# 1. Learning Curve with Phase Transitions
	# ========================================
	ax1 = fig.add_subplot(gs[0, :])

	unique_phases = sorted(set(phases))

	# Create phase segments using transition boundaries directly
	phase_segments = []
	for i, phase in enumerate(unique_phases):
		if i == 0:
			start_epoch = 1  # First epoch
		else:
			start_epoch = transitions[i-1]# + 1  # First epoch after previous transition
		
		if i < len(transitions):
			end_epoch = transitions[i]  # This transition ends the current phase
		else:
			end_epoch = max(epochs)  # Last epoch for final phase
		
		phase_segments.append((start_epoch, end_epoch, phase))

	for start_epoch, end_epoch, phase in phase_segments:
		ax1.axvspan(
			start_epoch,
			end_epoch,
			alpha=0.2,
			color=phase_colors[phase],
			label=f'Phase {phase}',
			zorder=0,
		)

	# Plot loss curves with enhanced styling
	ax1.plot(
		epochs, 
		train_losses, 
		color=train_loss_color,
		linestyle='-',
		linewidth=2.5, 
		label='Train',
		alpha=0.9, 
		marker='o', 
		markersize=2.5,
	)
	ax1.plot(
		epochs, 
		val_losses, 
		color=val_loss_color,
		linestyle='-',
		linewidth=2.5, 
		label='Validation', 
		alpha=0.9, 
		marker='o',
		markersize=2.5,
	)
	ymin, ymax = ax1.get_ylim()
	y_middle = (ymin + ymax) / 2.0

	for i, transition_epoch in enumerate(transitions):
		ax1.axvline(
			x=transition_epoch, 
			color=transition_color, 
			linestyle=':',
			linewidth=2.0,
			alpha=0.6,
			zorder=10,
		)
		
		if transition_epoch < len(val_losses):
			transition_loss = val_losses[transition_epoch]
			improvement_text = ""
			if transition_epoch > 0:
				prev_loss = val_losses[transition_epoch - 1]
				change = ((prev_loss - transition_loss) / prev_loss) * 100
				improvement_text = f" ({change:+.2f}%)"
			
			ax1.text(
				transition_epoch + 0.2,
				y_middle,
				f'T{i+1}{improvement_text}',
				rotation=90,
				fontsize=8,
				ha='left',
				va='center',
				color=transition_color,
				bbox=dict(
					boxstyle="round,pad=0.4",
					edgecolor='none',
					facecolor="#C5C5C5",
					alpha=0.5,
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
			label='Best',
			linewidth=1.5,
		)
	
	if early_stop_epoch is not None:
		ax1.axvline(
			x=early_stop_epoch, 
			color=early_stop_color, 
			linestyle=':',
			linewidth=1.5,
			alpha=0.95,
			zorder=12
		)
		ax1.text(
			early_stop_epoch + 0.25,
			y_middle,
			'Early Stopping', 
			rotation=90, 
			ha='left',
			va='center',
			fontsize=8,
			color=early_stop_color, 
		)
	
	ax1.set_xlabel('Epoch', fontsize=8)
	ax1.set_ylabel('Loss', fontsize=8)
	ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))

	ax1.set_title(f'Learning [Loss] Curve with Phase Transitions', fontsize=8, weight='bold')
	legend = ax1.legend(
		loc='upper right',
		fontsize=8,
		frameon=False,
		shadow=False,
		fancybox=False,
		edgecolor='none',
		facecolor='#FFFFFF',
		ncol=len(transitions)+5,
	)
	legend.set_zorder(20)
	ax1.grid(True, alpha=0.5)
	ax1.tick_params(axis='both', which='major', labelsize=8)
	ax1.set_xlim(left=0, right=max(epochs)+2)

	# ===================================
	# Hyperparameter Adaptation (LR + WD)
	# ===================================
	ax2 = fig.add_subplot(gs[1:, 1:])

	# Plot Learning Rate on primary axis
	for i in range(len(epochs) - 1):
		x0, x1 = epochs[i], epochs[i+1]
		y0, y1 = learning_rates[i], learning_rates[i+1]
		phase_color = phase_colors[phases[i+1]]
		ax2.plot(
			[x0, x1],
			[y0, y1],
			color=phase_color,
			linewidth=1.5,
			marker='o',
			markersize=2,
			alpha=0.5,
			linestyle='-'
		)

	# Create twin axis for Weight Decay
	ax2_twin = ax2.twinx()

	# Plot Weight Decay on secondary axis
	for i in range(len(epochs) - 1):
		x0, x1 = epochs[i], epochs[i+1]
		y0, y1 = weight_decays[i], weight_decays[i+1]
		phase_color = phase_colors[phases[i+1]]
		ax2_twin.plot(
			[x0, x1],
			[y0, y1],
			color=phase_color,
			linewidth=2.0,
			alpha=0.5,
			linestyle='--'  # Different line style for distinction
		)

	for transition_epoch in transitions:
		if transition_epoch < len(learning_rates):
			ax2.axvline(
				x=transition_epoch,
				color=transition_color,
				linewidth=2.0,
				alpha=0.6,
				linestyle=':',
				zorder=10
			)

	ax2.set_title('Hyperparameter Adaptation Across Phases\nLearning Rate (—) Weight Decay (--)', fontsize=8, weight='bold')
	ax2.set_ylabel('LR', fontsize=7, weight='bold')
	ax2_twin.set_ylabel('WD', fontsize=7, weight='bold')
	ax2.tick_params(axis='y', labelsize=7)
	ax2_twin.tick_params(axis='y', labelsize=7)
	ax2.grid(True, alpha=0.3, linestyle='-', color='#A3A3A3')
	ax2_twin.grid(True, alpha=0.5, linestyle='--', color="#8A8A8A")
	
	ax2.set_xlim(left=0, right=max(epochs)+1)
	ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
	ax2.set_xlabel('Epoch', fontsize=8, weight='bold')
	
	# ax2_twin.set_xlim(left=0, right=max(epochs)+1)
	
	# ==========================================
	# Phase Efficiency Analysis
	# ==========================================
	ax4 = fig.add_subplot(gs[1:, :1])
	phase_data = []
	for phase in unique_phases:
		phase_epochs = [e for e, p in zip(epochs, phases) if p == phase]
		metrics = calculate_advanced_phase_metrics(phase_epochs, val_losses, phase)
		if metrics:
			phase_data.append((phase, metrics))

	# Create more informative visualization
	if phase_data:
		phases_list = [p for p, _ in phase_data]
		
		# Choose primary metrics to display
		efficiencies = [m['efficiency'] for _, m in phase_data]  # Improvement per epoch
		convergence_qualities = [m['convergence_quality'] for _, m in phase_data]  # How consistent
		
		bars = ax4.bar(
			range(len(trainable_params_per_phase)), 
			trainable_params_per_phase,
			color=[phase_colors[p] for p in phases_list], 
			alpha=0.6,
			label='Trainable Parameters (%)', 
		)
		ax4.bar_label(
			bars,
			fmt='{:.2f}%',
			label_type='edge', 
			fontsize=7,
			padding=1,
			color=trainable_param_color,
		)
		
		# Twin axis: efficiency (more meaningful than raw improvement)
		ax4_twin = ax4.twinx()
		ax4_twin.plot(
			range(len(efficiencies)),
			efficiencies,
			linewidth=0.6,
			linestyle='-',
			marker='o',
			markersize=2.5,
			alpha=0.6,
			color=loss_imp_color,
			label='Efficiency (%/ep)',
		)
		
		# Add convergence quality as scatter size or color intensity
		for i, (efficiency, convergence) in enumerate(zip(efficiencies, convergence_qualities)):
			ax4_twin.text(
				i, 
				efficiency * 1.05 if efficiency > 0 else efficiency * 0.8,
				f'{efficiency:.3f} (R²: {convergence:.3f})',
				ha='center', 
				va='bottom', #if efficiency > 0 else 'top',
				fontsize=7,
				color=loss_imp_color,
			)

	ax4.set_xlabel('Phase', fontsize=8, weight='bold')
	ax4.set_ylabel(f'Trainable Parameters (%)\nTotal: {total_model_params:,}', fontsize=8, weight='bold', color=trainable_param_color)
	ax4_twin.set_ylabel('Learning Efficiency (%/ep)', fontsize=8, color=loss_imp_color)
	ax4.set_title('Phase Efficiency Analysis', fontsize=8, weight='bold')
	
	phase_labels = [f'{p}' for p in phases_list]
	ax4.set_xticklabels(phase_labels)
	ax4.set_xticks(range(len(phase_labels)))
	ax4.grid(axis='y', alpha=0.25, color="#A3A3A3")

	ax4.tick_params(axis='y', labelcolor=trainable_param_color, labelsize=8)
	ax4_twin.tick_params(axis='y', labelcolor=loss_imp_color, labelsize=8)

	# Match spine colors with their labels
	ax4.spines['left'].set_color(trainable_param_color)
	ax4_twin.spines['right'].set_color(loss_imp_color)

	ax4_twin.spines['top'].set_visible(False)
	ax4.spines['top'].set_visible(False)

	plt.suptitle(
		f'Progressive Layer Unfreezing\nPhase Transition Analysis', 
		fontsize=11,
		weight='bold',
	)
	
	plt.savefig(
		file_path, 
		dpi=300, 
		bbox_inches='tight', 
		facecolor='white', 
		edgecolor='none',
	)
	
	plt.close()

	# ====================================
	# Statistics and Insights
	# ====================================
	total_epochs = len(epochs)
	num_phases = len(set(phases))

	# Calculate advanced phase metrics
	advanced_phase_data = []
	for phase in unique_phases:
		phase_epochs = [e for e, p in zip(epochs, phases) if p == phase]
		metrics = calculate_advanced_phase_metrics(phase_epochs, val_losses, phase)
		if metrics:
			advanced_phase_data.append((phase, metrics))

	# Basic performance metrics
	total_improvement = ((val_losses[0] - min(val_losses)) / val_losses[0] * 100) if val_losses and val_losses[0] > 0 else 0
	final_train_loss = train_losses[-1]
	final_val_loss = val_losses[-1]
	best_val_loss = min(val_losses) if val_losses else 0.0

	# Advanced diagnostics
	loss_divergence = ((final_val_loss - final_train_loss) / final_val_loss * 100) if final_val_loss > 0 else 0.0
	performance_delta = ((final_val_loss - best_val_loss) / best_val_loss * 100) if best_val_loss > 0 else 0.0

	# Phase effectiveness analysis using advanced metrics
	if advanced_phase_data:
		efficiencies = [m['efficiency'] for _, m in advanced_phase_data]
		convergence_qualities = [m['convergence_quality'] for _, m in advanced_phase_data]
		stabilities = [m['stability'] for _, m in advanced_phase_data]
		
		# Find most effective phase by efficiency
		most_efficient_idx = np.argmax(efficiencies) if efficiencies else 0
		least_efficient_idx = np.argmin(efficiencies) if efficiencies else 0
		most_efficient_phase = f"Phase {advanced_phase_data[most_efficient_idx][0]} ({efficiencies[most_efficient_idx]:.3f}%/epoch)"
		least_efficient_phase = f"Phase {advanced_phase_data[least_efficient_idx][0]} ({efficiencies[least_efficient_idx]:.3f}%/epoch)"
		
		# Find most stable phase
		most_stable_idx = np.argmax(stabilities) if stabilities else 0
		most_stable_phase = f"Phase {advanced_phase_data[most_stable_idx][0]} (Stability: {stabilities[most_stable_idx]:.2f})"
		
		avg_efficiency = np.mean(efficiencies) if efficiencies else 0
		avg_convergence = np.mean(convergence_qualities) if convergence_qualities else 0
	else:
		most_efficient_phase = "N/A"
		least_efficient_phase = "N/A"
		most_stable_phase = "N/A"
		avg_efficiency = 0
		avg_convergence = 0

	# Phase transition effectiveness (existing logic)
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

	# Training efficiency metrics
	if total_epochs > 0:
		total_efficiency = total_improvement / total_epochs
		time_to_best = (best_epoch + 1) if best_epoch is not None else total_epochs
		efficiency_to_best = total_improvement / time_to_best if time_to_best > 0 else 0
	else:
		total_efficiency = 0
		efficiency_to_best = 0

	# Best model context
	best_model_phase = phases[best_epoch] if best_epoch is not None else -1
	trainable_info_at_best = f"(Phase {best_model_phase})"

	# Generate comprehensive summary
	summary_text = f"""
	COMPREHENSIVE TRAINING ANALYSIS:

	OVERALL PERFORMANCE:
			• Total Epochs: {total_epochs}
			• Number of Phases: {num_phases}
			• Final Training Loss: {final_train_loss:.4f}
			• Final Validation Loss: {final_val_loss:.4f}
			• Best Validation Loss: {best_val_loss:.4f}
			• Total Improvement: {total_improvement:.2f}%
			• Overall Efficiency: {total_efficiency:.3f}% per epoch
			• Efficiency to Best: {efficiency_to_best:.3f}% per epoch
			• Training Status: {'Early Stopped' if early_stop_epoch else 'Completed'}

	DIAGNOSTICS:
			• Loss Divergence (Train vs Val): {loss_divergence:.1f}% {'[OVERFITTING RISK]' if loss_divergence > 20 else '[OK]'}
			• Performance Delta (Best vs Final): {performance_delta:.1f}% {'[OVERTRAINING RISK]' if performance_delta > 5 else '[OK]'}
			• Best Model: Epoch {best_epoch + 1 if best_epoch is not None else 'N/A'} {trainable_info_at_best}

	PHASE EFFECTIVENESS ANALYSIS:
			• Average Learning Efficiency: {avg_efficiency:.3f}% per epoch
			• Average Convergence Quality (R²): {avg_convergence:.3f}
			• Most Efficient Phase: {most_efficient_phase}
			• Least Efficient Phase: {least_efficient_phase}
			• Most Stable Phase: {most_stable_phase}

	TRANSITION ANALYSIS:
			• Total Transitions: {len(transitions)}
			• Average Improvement per Transition: {avg_transition_improvement:.2f}%
			• Transition Success Rate: {len([x for x in transition_improvements if x > 0])}/{len(transition_improvements)} positive

	HYPERPARAMETER ADAPTATION:
			• Learning Rate Range: {min(learning_rates):.2e} → {max(learning_rates):.2e}
			• Weight Decay Range: {min(weight_decays):.2e} → {max(weight_decays):.2e}
			• LR Reduction Factor: {(learning_rates[0]/learning_rates[-1]):.1f}x
	"""

	if transitions:
		summary_text += f"\n    TRANSITION EPOCHS: {transitions}"

	# Detailed phase insights
	phase_insights = "\n    DETAILED PHASE ANALYSIS:\n"
	for phase, metrics in advanced_phase_data:
		phase_insights += (
			f"    • Phase {phase}: {metrics['duration']} epochs\n"
			f"      ├─ Efficiency: {metrics['efficiency']:+.3f}%/epoch\n"
			f"      ├─ Robust Improvement: {metrics['robust_improvement']:+.2f}%\n"
			f"      ├─ Convergence Quality (R²): {metrics['convergence_quality']:.3f}\n"
			f"      ├─ Volatility (CV): {metrics['volatility']:.3f}\n"
			f"      ├─ Early vs Late Learning: {metrics['early_vs_late']:+.2f}%\n"
			f"      └─ Final Loss: {metrics['final_loss']:.4f}\n\n"
		)

	summary_text += phase_insights

	print(summary_text)

	# Updated analysis results for return
	analysis_results = {
		'total_improvement': total_improvement,
		'total_efficiency': total_efficiency,
		'efficiency_to_best': efficiency_to_best,
		'num_transitions': len(transitions),
		'most_efficient_phase': advanced_phase_data[most_efficient_idx][0] if advanced_phase_data else 0,
		'avg_efficiency': avg_efficiency,
		'avg_convergence_quality': avg_convergence,
		'transition_improvements': transition_improvements,
		'transition_success_rate': len([x for x in transition_improvements if x > 0]) / len(transition_improvements) if transition_improvements else 0,
		'lr_adaptation_factor': learning_rates[0]/learning_rates[-1] if learning_rates[-1] > 0 else 1.0,
		'loss_divergence': loss_divergence,
		'performance_delta': performance_delta,
		'advanced_phase_metrics': {p: m for p, m in advanced_phase_data}
	}

	return analysis_results

def collect_progressive_training_history(
		training_losses: List[float],
		in_batch_metrics_all_epochs: List[Dict],
		learning_rates: List[float],
		weight_decays: List[float],
		phases: List[int],
		trainable_params_per_phase: List[int],
		total_model_params: int,
		embedding_drifts: List[float],
		phase_transitions: List[int],
		early_stop_epoch: Optional[int] = None,
		best_epoch: Optional[int] = None
	) -> Dict:
	
	val_losses = [metrics.get('val_loss', 0.0) for metrics in in_batch_metrics_all_epochs]
	epochs = list(range(len(training_losses)))
	
	return {
		'epochs': epochs,
		'train_losses': training_losses,
		'val_losses': val_losses,
		'learning_rates': learning_rates,
		'weight_decays': weight_decays,
		'phases': phases,
		'trainable_params_per_phase': trainable_params_per_phase,
		'total_model_params': total_model_params,
		'embedding_drifts': embedding_drifts,
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
		# strategy_colors = {'full': '#0058a5', 'lora': '#f58320be', 'progressive': '#cc40df'}  # Blue, Orange, Purple
		# pretrained_colors = {'ViT-B/32': '#745555', 'ViT-B/16': '#9467bd', 'ViT-L/14': '#e377c2', 'ViT-L/14@336px': '#696969'}
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
	# strategy_colors = {'full': '#0058a5', 'lora': '#f58320be', 'progressive': '#cc40df'}  # Blue, Orange, Green
	# pretrained_colors = {'ViT-B/32': '#745555', 'ViT-B/16': '#9467bd', 'ViT-L/14': '#e377c2', 'ViT-L/14@336px': '#696969'}

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
	
	# strategy_colors = {'full': '#0058a5', 'lora': '#f58320be', 'progressive': '#cc40df'}
	# pretrained_colors = {'ViT-B/32': '#745555', 'ViT-B/16': '#9467bd', 'ViT-L/14': '#e377c2', 'ViT-L/14@336px': '#696969'}
	
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
					fill="#0205B3",  # Different color to distinguish from score
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
					fname = (
						f"{dataset_name}_"
						f"{'_'.join(finetune_strategies)}_"
						f"finetuned_vs_CLIP_"
						f"{re.sub(r'[/@]', '-', model_name)}_"
						f"{mode.replace('-', '_')}_"
						f"{metric}_"
						f"comparison.png")
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
		figure_size=(11, 10),
		DPI: int = 250,
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
	print(f"{len(finetune_strategies)} Finetune strategies: {finetune_strategies}")

	if not finetune_strategies:
		print("WARNING: No valid finetune strategies provided. Skipping...")
		return
		
	for mode in modes:
		pretrained_dict = pretrained_img2txt_dict if mode == "Image-to-Text" else pretrained_txt2img_dict
		finetuned_dict = finetuned_img2txt_dict if mode == "Image-to-Text" else finetuned_txt2img_dict
		for metric in metrics:
			# Create figure with slightly adjusted size for better annotation spacing
			fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
			
			# Create filename for the output
			fname = (
				f"{dataset_name}_"
				f"{'_'.join(finetune_strategies)}_"
				f"finetuned_vs_CLIP_"
				f"{re.sub(r'[/@]', '-', model_name)}_"
				f"{mode.replace('-', '_')}_"
				f"{metric}_"
				f"comparison.png"
			)
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
				linewidth=4.5,
				markersize=6.5,
				alpha=0.98,
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
					linewidth=2.0,
					markersize=4,
					alpha=0.75,
				)
			
			# Analyze plot data to place annotations intelligently
			key_k_values = [1, 10, 20]  # These are the key points to annotate
			annotation_positions = {}    # To store planned annotation positions
			
			# First pass: gather data about values and improvements
			for k in key_k_values:
				if k in k_values:
					k_idx = k_values.index(k)
					pre_val = pretrained_vals[k_idx]
					finetuned_vals_at_k = {
						strategy: finetuned_dict[model_name][strategy][metric].get(str(k), float('nan')) 
						for strategy in finetune_strategies
					}
					
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
				best_text_color = positive_pct_col if best_imp >= 0 else negative_pct_col
				best_arrow_style = '<|-' if best_imp >= 0 else '-|>'
				
				# For worst annotation (typically placed below)
				worst_text_color = positive_pct_col if worst_imp >= 0 else negative_pct_col
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
					bbox=dict(
						facecolor='#ffffff', 
						edgecolor='none', 
						alpha=0.7, 
						pad=0.1
					),
					arrowprops=dict(
						arrowstyle=best_arrow_style,
						color=best_text_color,
						shrinkA=0,
						shrinkB=3,
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
						bbox=dict(
							facecolor='#ffffff', 
							edgecolor='none', 
							alpha=0.7, 
							pad=0.1
						),
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
			ax.set_ylim(min(0, y_min - padding), max(1, y_max + padding))
			# ax.set_ylim(max(-0.02, y_min - padding), min(1.02, y_max + padding))
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
		models_dict: dict,
		figure_size=(16, 8),
		DPI: int = 200,
	):
	metrics = ["mP", "mAP", "Recall"]
	modes = ["Image-to-Text", "Text-to-Image"]
	all_model_architectures = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
	# Validate model_name and finetune_strategies
	print(f"{len(finetune_strategies)} Finetune strategies: {finetune_strategies}")

	if not finetune_strategies:
		print("WARNING: No valid finetune strategies provided. Skipping...")
		return

	for mode in modes:
		finetuned_dict = finetuned_img2txt_dict if mode == "Image-to-Text" else finetuned_txt2img_dict
		if model_name not in finetuned_dict or not all(strategy in finetuned_dict.get(model_name, {}) for strategy in finetune_strategies):
			print(f"WARNING: Some strategies for {model_name} not found in finetuned_{mode.lower().replace('-', '_')}_dict. Skipping...")
			return

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
				k
				for k in topK_values
				if str(k) in pretrained_dict.get(model_name, {}).get(metric, {})
			)
			for strategy in finetune_strategies:
				k_values = sorted(
					set(k_values) & set(int(k)
					for k in finetuned_dict.get(model_name, {}).get(strategy, {}).get(metric, {}).keys())
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
						text_color = positive_pct_col if best_imp >= 0 else negative_pct_col 
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
						text_color = negative_pct_col if worst_imp <= 0 else positive_pct_col
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



	def calculate_improvements(mode, metric, strategy):
		"""Helper function to calculate improvements for a given mode, metric, and strategy"""
		pretrained_dict = pretrained_img2txt_dict if mode == "Image-to-Text" else pretrained_txt2img_dict
		finetuned_dict = finetuned_img2txt_dict if mode == "Image-to-Text" else finetuned_txt2img_dict
		
		improvements = []
		absolute_improvements = []
		
		for k in topK_values:
			if (str(k) in pretrained_dict.get(model_name, {}).get(metric, {}) and 
				str(k) in finetuned_dict.get(model_name, {}).get(strategy, {}).get(metric, {})):
				
				pre_val = pretrained_dict[model_name][metric][str(k)]
				fine_val = finetuned_dict[model_name][strategy][metric][str(k)]
				
				if pre_val != 0:
					rel_imp = (fine_val - pre_val) / pre_val * 100
					abs_imp = fine_val - pre_val
					improvements.append(rel_imp)
					absolute_improvements.append(abs_imp)
		
		return improvements, absolute_improvements

	def enhanced_statistical_analysis():
		print(f"ENHANCED QUANTITATIVE ANALYSIS WITH STATISTICAL SIGNIFICANCE".center(150, "="))
		
		results_summary = {}
		
		for mode in modes:
			print(f"\nMode: {mode}")
			print(f"{'-'*60}")
			results_summary[mode] = {}
			
			for metric in metrics:
				print(f"\n{metric} Analysis:")
				results_summary[mode][metric] = {}
				
				for strategy in finetune_strategies:
					# if strategy == "Probe":  # Skip probe as it's baseline
					# 	continue
					
					improvements, abs_improvements = calculate_improvements(mode, metric, strategy)
					
					if improvements:
						# Calculate comprehensive statistics
						mean_improvement = np.mean(improvements)
						std_improvement = np.std(improvements)
						median_improvement = np.median(improvements)
						
						# Effect size (Cohen's d equivalent)
						effect_size = mean_improvement / std_improvement if std_improvement > 0 else 0
						
						# Calculate consistency (percentage of positive improvements)
						positive_improvements = sum(1 for x in improvements if x > 0)
						consistency = (positive_improvements / len(improvements)) * 100
						
						# Statistical significance test (one-sample t-test against 0)
						if len(improvements) > 1:
							t_stat, p_value = scipy.stats.ttest_1samp(improvements, 0)
							is_significant = p_value < 0.05
						else:
							t_stat, p_value, is_significant = 0, 1, False
						
						# Store results
						results_summary[mode][metric][strategy] = {
							'mean': mean_improvement,
							'std': std_improvement,
							'median': median_improvement,
							'effect_size': effect_size,
							'consistency': consistency,
							'p_value': p_value,
							'significant': is_significant,
							'min': min(improvements),
							'max': max(improvements)
						}
						
						print(f"\t{strategy.capitalize()}:")
						print(f"\t\tMean improvement: {mean_improvement:+.2f}% ± {std_improvement:.2f}%")
						print(f"\t\tMedian improvement: {median_improvement:+.2f}%")
						print(f"\t\tEffect size: {effect_size:.3f}")
						print(f"\t\tConsistency: {consistency:.1f}% of K values improved")
						print(f"\t\tRange: [{min(improvements):+.2f}%, {max(improvements):+.2f}%]")
						print(f"\t\tStatistical significance: {'Yes' if is_significant else 'No'} (p={p_value:.4f})")
		
		return results_summary

	def k_value_distribution_analysis():
		"""Analyze performance across different K value ranges"""
		print(f"\n{'='*80}")
		print(f"K-VALUE DISTRIBUTION ANALYSIS")
		print(f"{'='*80}")
		
		k_categories = {
			"Low-K (1-3)": [k for k in [1, 3] if k in topK_values],
			"Mid-K (5-10)": [k for k in [5, 10] if k in topK_values], 
			"High-K (15-20)": [k for k in [15, 20] if k in topK_values]
		}
		
		for mode in modes:
			print(f"\nMode: {mode}")
			for category, k_vals in k_categories.items():
				if not k_vals:
					continue
				print(f"\n{category}:")
				
				for strategy in finetune_strategies:
					if strategy == "Probe":
						continue
					
					category_improvements = []
					for metric in metrics:
						for k in k_vals:
							improvements, _ = calculate_improvements(mode, metric, strategy)
							if improvements and k in topK_values:
								k_idx = topK_values.index(k) if k in topK_values else None
								if k_idx is not None and k_idx < len(improvements):
									category_improvements.append(improvements[k_idx])
					
					if category_improvements:
						avg_cat_improvement = np.mean(category_improvements)
						std_cat_improvement = np.std(category_improvements)
						print(f"  {strategy.capitalize()}: {avg_cat_improvement:+.2f}% ± {std_cat_improvement:.2f}%")

	def asymmetry_analysis():
		"""Quantify cross-modal directional bias"""
		print(f"\n{'='*80}")
		print(f"CROSS-MODAL ASYMMETRY ANALYSIS")
		print(f"{'='*80}")
		
		for strategy in finetune_strategies:
			if strategy == "Probe":
				continue
			
			# Calculate average improvements for each direction
			i2t_improvements = []
			t2i_improvements = []
			
			for metric in metrics:
				i2t_imp, _ = calculate_improvements("Image-to-Text", metric, strategy)
				t2i_imp, _ = calculate_improvements("Text-to-Image", metric, strategy)
				i2t_improvements.extend(i2t_imp)
				t2i_improvements.extend(t2i_imp)
			
			if i2t_improvements and t2i_improvements:
				i2t_avg = np.mean(i2t_improvements)
				t2i_avg = np.mean(t2i_improvements)
				
				# Asymmetry ratio
				asymmetry_ratio = t2i_avg / i2t_avg if i2t_avg != 0 else float('inf')
				
				print(f"\n{strategy.capitalize()} Fine-tuning:")
				print(f"  Image-to-Text average improvement: {i2t_avg:+.2f}%")
				print(f"  Text-to-Image average improvement: {t2i_avg:+.2f}%")
				print(f"  Asymmetry ratio (T2I/I2T): {asymmetry_ratio:.2f}x")
				print(f"  Directional bias: {'Text-to-Image favored' if asymmetry_ratio > 1.5 else 'Balanced' if 0.67 <= asymmetry_ratio <= 1.5 else 'Image-to-Text favored'}")

	def get_model_parameter_counts(models_dict, finetune_strategies):
		"""
		Dynamically extract parameter counts from actual model instances
		
		Returns:
			dict: Parameter counts for each strategy
		"""
		param_counts = {}
		
		if models_dict is None:
			print("Warning: No model instances provided. Using estimated parameter counts.")
			return None
		
		for strategy in finetune_strategies:
			strategy_key = strategy.lower()
			
			if strategy_key not in models_dict:
				print(f"Warning: Model for strategy '{strategy}' not found in models_dict")
				continue
			
			model = models_dict[strategy_key]
			
			# Count total parameters
			total_params = sum(p.numel() for p in model.parameters())
			
			# Count trainable parameters
			trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
			
			# Count frozen parameters
			frozen_params = total_params - trainable_params
			
			param_counts[strategy] = {
				'total': total_params,
				'trainable': trainable_params,
				'frozen': frozen_params,
				'trainable_percentage': (trainable_params / total_params) * 100 if total_params > 0 else 0
			}
			
			print(f"Model parameter analysis for {strategy.capitalize()}:")
			print(f"  Total parameters: {total_params:,}")
			print(f"  Trainable parameters: {trainable_params:,} ({param_counts[strategy]['trainable_percentage']:.2f}%)")
			print(f"  Frozen parameters: {frozen_params:,}")
		
		# Also get pretrained model params for reference
		if 'pretrained' in models_dict:
			pretrained_model = models_dict['pretrained']
			pretrained_total = sum(p.numel() for p in pretrained_model.parameters())
			param_counts['pretrained'] = {
				'total': pretrained_total,
				'trainable': pretrained_total,  # All params trainable in original model
				'frozen': 0,
				'trainable_percentage': 100.0
			}
		
		return param_counts

	def efficiency_analysis():
		"""Compare computational efficiency across strategies using actual parameter counts"""
		print(f"\n{'='*80}")
		print(f"COMPUTATIONAL EFFICIENCY ANALYSIS (DYNAMIC PARAMETER COUNTING)")
		print(f"{'='*80}")
		
		# Get actual parameter counts from model instances
		param_counts = get_model_parameter_counts(models_dict, finetune_strategies)
		
		if param_counts is None:
			# Fallback to estimated values if no models provided
			print("Falling back to estimated parameter counts...")
			model_params = {
				"ViT-B/32": 151277313,
				"ViT-B/16": 149620737,
				"ViT-L/14": 427616513,
				"RN50": 102010368,
				"RN101": 119547648,
				"RN50x4": 319286272,
				"RN50x16": 955375616,
				"RN50x64": 3632465920,
			}
			total_params = model_params.get(model_name, 150000000)
			
			# Use estimated efficiency metrics
			efficiency_metrics = {
				"full": {"trainable_params": total_params, "relative_cost": 1.0, "memory_factor": 1.0},
				"lora": {"trainable_params": int(total_params * 0.01), "relative_cost": 0.15, "memory_factor": 0.3},
				"progressive": {"trainable_params": total_params, "relative_cost": 1.2, "memory_factor": 1.1},
			}
		else:
			# Use actual parameter counts
			efficiency_metrics = {}
			for strategy in finetune_strategies:
				strategy_key = strategy.lower()
				if strategy_key in param_counts:
					params_data = param_counts[strategy_key]
					
					# Estimate relative costs based on trainable parameter ratio
					trainable_ratio = params_data['trainable_percentage'] / 100.0
					
					# Base relative costs on complexity
					if strategy_key == "full":
						relative_cost = 1.0
						memory_factor = 1.0
					elif strategy_key == "lora":
						relative_cost = 0.1 + (trainable_ratio * 0.5)  # Base + scaling factor
						memory_factor = 0.3 + (trainable_ratio * 0.4)
					elif strategy_key == "progressive":
						relative_cost = 1.0 + 0.3  # Overhead for phase management
						memory_factor = 1.1
					elif strategy_key == "probe":
						relative_cost = 0.05  # Very minimal training
						memory_factor = 0.1
					else:
						relative_cost = trainable_ratio
						memory_factor = 0.5 + (trainable_ratio * 0.5)
					
					efficiency_metrics[strategy_key] = {
						"trainable_params": params_data['trainable'],
						"total_params": params_data['total'],
						"relative_cost": relative_cost,
						"memory_factor": memory_factor,
						"trainable_percentage": params_data['trainable_percentage']
					}
		
		# Calculate and display efficiency metrics
		print(f"\nEfficiency Analysis Results:")
		print(f"{'-'*50}")
		
		efficiency_results = {}
		
		for strategy in finetune_strategies:
			strategy_key = strategy.lower()
			if strategy_key == "probe" or strategy_key not in efficiency_metrics:
				continue
			
			metrics_data = efficiency_metrics[strategy_key]
			
			# Calculate overall average improvement
			all_improvements = []
			for mode in modes:
				for metric in metrics:
					improvements, _ = calculate_improvements(mode, metric, strategy)
					all_improvements.extend(improvements)
			
			if all_improvements:
				avg_improvement = np.mean(all_improvements)
				std_improvement = np.std(all_improvements)
				trainable_params = metrics_data["trainable_params"]
				total_params = metrics_data.get("total_params", trainable_params)
				
				params_millions = trainable_params / 1e6
				total_params_millions = total_params / 1e6
				
				# Calculate efficiency scores
				efficiency_score = avg_improvement / params_millions if params_millions > 0 else 0
				cost_efficiency = avg_improvement / metrics_data["relative_cost"] if metrics_data["relative_cost"] > 0 else 0
				
				# Store results
				efficiency_results[strategy] = {
					'avg_improvement': avg_improvement,
					'std_improvement': std_improvement,
					'trainable_params_m': params_millions,
					'total_params_m': total_params_millions,
					'efficiency_score': efficiency_score,
					'cost_efficiency': cost_efficiency,
					'relative_cost': metrics_data["relative_cost"],
					'memory_factor': metrics_data["memory_factor"],
					'trainable_percentage': metrics_data.get("trainable_percentage", 100.0)
				}
				
				print(f"\n{strategy.capitalize()} Fine-tuning:")
				print(f"  Total parameters: {total_params_millions:.1f}M")
				print(f"  Trainable parameters: {params_millions:.1f}M ({metrics_data.get('trainable_percentage', 100.0):.1f}% of total)")
				print(f"  Relative training cost: {metrics_data['relative_cost']:.2f}x")
				print(f"  Memory overhead: {metrics_data['memory_factor']:.1f}x")
				print(f"  Overall average improvement: {avg_improvement:+.2f}% ± {std_improvement:.2f}%")
				print(f"  Performance/Parameter efficiency: {efficiency_score:.4f}%/M trainable params")
				print(f"  Performance/Cost efficiency: {cost_efficiency:.2f}%/cost unit")
		
		# Summary comparison
		if len(efficiency_results) > 1:
			print(f"\n{'='*50}")
			print(f"EFFICIENCY RANKING SUMMARY")
			print(f"{'='*50}")
			
			# Rank by different criteria
			rankings = {
				'Performance/Parameter': sorted(efficiency_results.items(), key=lambda x: x[1]['efficiency_score'], reverse=True),
				'Performance/Cost': sorted(efficiency_results.items(), key=lambda x: x[1]['cost_efficiency'], reverse=True),
				'Overall Performance': sorted(efficiency_results.items(), key=lambda x: x[1]['avg_improvement'], reverse=True),
				'Parameter Efficiency': sorted(efficiency_results.items(), key=lambda x: x[1]['trainable_params_m'])
			}
			
			for criterion, ranking in rankings.items():
				print(f"\nBest {criterion}:")
				for i, (strategy, data) in enumerate(ranking[:3], 1):
					if criterion == 'Performance/Parameter':
						value = f"{data['efficiency_score']:.4f}%/M"
					elif criterion == 'Performance/Cost':
						value = f"{data['cost_efficiency']:.2f}%/cost"
					elif criterion == 'Overall Performance':
						value = f"{data['avg_improvement']:+.2f}%"
					else:  # Parameter Efficiency
						value = f"{data['trainable_params_m']:.1f}M params"
					
					print(f"  {i}. {strategy.capitalize()}: {value}")
		
		return efficiency_results

	def failure_mode_analysis():
		"""Identify limitations and failure modes of each strategy"""
		print(f"\n{'='*80}")
		print(f"FAILURE MODE AND LIMITATION ANALYSIS")
		print(f"{'='*80}")
		
		for strategy in finetune_strategies:
			if strategy == "Probe":
				continue
			
			print(f"\n{strategy.capitalize()} Strategy Analysis:")
			
			# Find worst-performing metric combinations
			worst_performances = []
			
			for mode in modes:
				for metric in metrics:
					improvements, _ = calculate_improvements(mode, metric, strategy)
					if improvements:
						avg_improvement = np.mean(improvements)
						worst_performances.append((f"{mode} {metric}", avg_improvement))
			
			# Sort by performance (worst first)
			worst_performances.sort(key=lambda x: x[1])
			
			print("  Weakest performance areas:")
			for i, (metric_combo, improvement) in enumerate(worst_performances[:3]):
				print(f"    {i+1}. {metric_combo}: {improvement:+.2f}%")
			
			# Strategy-specific observations
			strategy_insights = {
				"Progressive": "Notable: U-shaped learning curve with initial performance drop",
				"LoRA": "Notable: Conservative improvements but better parameter efficiency",
				"Full": "Notable: Strong T2I gains but potential I2T degradation over time"
			}
			
			if strategy in strategy_insights:
				print(f"  Key characteristic: {strategy_insights[strategy]}")

	def practical_recommendations():
		"""Generate actionable recommendations"""
		print(f"\n{'='*80}")
		print(f"PRACTICAL RECOMMENDATIONS FOR DEPLOYMENT")
		print(f"{'='*80}")
		
		print("\nStrategy Selection Guidelines:")
		
		# Determine best strategy for each use case
		best_i2t_strategy = None
		best_t2i_strategy = None
		best_balanced_strategy = None
		
		strategy_scores = {}
		for strategy in finetune_strategies:
			if strategy == "Probe":
				continue
			
			i2t_scores = []
			t2i_scores = []
			
			for metric in metrics:
				i2t_imp, _ = calculate_improvements("Image-to-Text", metric, strategy)
				t2i_imp, _ = calculate_improvements("Text-to-Image", metric, strategy)
				
				if i2t_imp:
					i2t_scores.append(np.mean(i2t_imp))
				if t2i_imp:
					t2i_scores.append(np.mean(t2i_imp))
			
			if i2t_scores and t2i_scores:
				strategy_scores[strategy] = {
					'i2t': np.mean(i2t_scores),
					't2i': np.mean(t2i_scores),
					'balanced': (np.mean(i2t_scores) + np.mean(t2i_scores)) / 2
				}
		
		if strategy_scores:
			best_i2t_strategy = max(strategy_scores, key=lambda x: strategy_scores[x]['i2t'])
			best_t2i_strategy = max(strategy_scores, key=lambda x: strategy_scores[x]['t2i'])
			best_balanced_strategy = max(strategy_scores, key=lambda x: strategy_scores[x]['balanced'])
		
		print(f"• Best for Image-to-Text tasks: {best_i2t_strategy.capitalize() if best_i2t_strategy else 'N/A'}")
		print(f"• Best for Text-to-Image tasks: {best_t2i_strategy.capitalize() if best_t2i_strategy else 'N/A'}")
		print(f"• Best balanced performance: {best_balanced_strategy.capitalize() if best_balanced_strategy else 'N/A'}")
		
		print("\nResource Considerations:")
		print("• Limited GPU memory: LoRA fine-tuning")
		print("• Limited training time: LoRA fine-tuning")
		print("• Maximum performance: Progressive unfreezing (with sufficient time)")
		print("• Quick adaptation: Full fine-tuning with early stopping")
		
		print("\nDomain-Specific Recommendations:")
		print("• Historical document retrieval: Progressive (handles domain shift well)")
		print("• General-purpose applications: LoRA (maintains broader capabilities)")
		print("• Content generation systems: Full fine-tuning (strong T2I performance)")

	print(f"OVERALL PERFORMANCE SUMMARY [QUANTITATIVE ANALYSIS]".center(160, "="))

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

	results_summary = enhanced_statistical_analysis()
	k_value_distribution_analysis()
	asymmetry_analysis()
	efficiency_analysis()
	failure_mode_analysis()
	practical_recommendations()
	print(f"{'='*80}\n")

	return results_summary

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
		DPI: int = 200,
		figure_size: Tuple[int, int] = (11, 6),
) -> None:
		
		num_epochs = len(train_losses)
		if num_epochs <= 1:
			return

		epochs = np.arange(1, num_epochs + 1)

		# For readability we show at most 20 x‑ticks
		num_xticks = min(20, num_epochs)
		selective_xticks = np.linspace(1, num_epochs, num_xticks, dtype=int)

		colors = {
			"train": "#1f77b4",
			"val": "#c75f03",
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

		# ---------------
		# 1️⃣ Loss curve
		# ---------------
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