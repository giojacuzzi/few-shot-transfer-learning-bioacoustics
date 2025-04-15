##########################################################################################
# Plot segment level performance comparisons between pre-trained source and custom target models
#
# Input:
# - Threshold performance results at "results/{model_stub}/test/segment_perf/threshold_perf_{model_tag}"
# - Figure labels table ("data/figures/fig_labels.csv")
# - Site presence/absence table ("data/test/site_presence_absence.csv")
#
# Output:
# - Plots for precision-recall threshold, AUC, and confidence score distribution (Figs 4, A.1, A.2)
#
# User-defined parameters
model_stub = 'OESF_1.0'
labels_to_plot = c("sooty grouse", "marbled murrelet", "golden-crowned kinglet", "belted kingfisher", "black-throated gray warbler", "wilson's warbler")
##########################################################################################

library(dplyr)
library(tools)
library(ggplot2)
library(cowplot)
library(patchwork)
library(stringr)
source('src/figures/global.R')

path_source = paste('results/', model_stub, '/test/segment_perf/threshold_perf_source', sep='')
path_target = paste('results/', model_stub, '/test/segment_perf/threshold_perf_target', sep='')

fig_labels = read.csv('data/figures/fig_labels.csv')

load_perf = function(path, model_tag) {
  files = list.files(path = path, pattern = "\\.csv$", full.names = TRUE)
  perf = lapply(files, function(file) {
    label = file_path_sans_ext(basename(file))
    data = read.csv(file)
    data$label = label
    data$model = model_tag
    # Add missing values
    data = rbind(data.frame(threshold = 0.0, precision = 0.0, recall = 1.0, label = label, model = model_tag), data)
    data = rbind(data, data.frame(threshold = 1.0, precision = 1.0, recall = 0.0, label = label, model = model_tag))
    return(data)
  })
  perf = bind_rows(perf)
  return(perf)
}

conf_to_logit = function(p) {
  return(log(p / (1 - p)))
}

# Combine performance metrics for both source and target models
perf_source = load_perf(path_source, 'source')
perf_source$f1 = 2*perf_source$recall * perf_source$precision/(perf_source$recall+perf_source$precision)

perf_target     = load_perf(path_target, 'target')
perf_target$f1 = 2*perf_target$recall * perf_target$precision/(perf_target$recall+perf_target$precision)

perf = bind_rows(perf_source, perf_target)
perf = perf %>% filter(!str_detect(label, paste(c(labels_to_remove), collapse = "|")))
perf$label = factor(str_to_title(perf$label))
perf$model = factor(perf$model, levels = c('source', 'target'))
perf$model = recode(perf$model, "source" = "Source", "target" = "Target")
perf$logit = conf_to_logit(perf$threshold)

perf_present = perf[perf$label %in% str_to_title(label_counts[label_counts$count > 0, 'label']), ]
perf_selected_species = perf[perf$label %in% str_to_title(labels_to_plot), ]

# Figure A.1: Differences in precision, recall, and F1 performance across decision thresholds between source (red) and target (blue) models for all present classes.
plot_threshold_pr = ggplot(perf_present, aes(x = threshold)) +
  geom_path(aes(y = recall, linetype = "Recall", color = model)) +
  geom_path(aes(y = precision, linetype = "Precision", color = model)) +
  geom_path(aes(y = f1, linetype = "F1", color = model)) +
  facet_wrap(~ str_trunc(as.character(label), width=24), ncol = 7, scales = "free_y") +
  scale_color_manual(values = c("Target" = "royalblue", "Source" = "salmon")) +
  scale_linetype_manual(values = c("Recall" = "dashed", "Precision" = "solid", "F1" = "dotted")) +
  labs(title = "Segment level test performance", x = "Threshold", y = "Performance", color = 'Model', linetype = 'Metric') +
  theme_bw() + theme(aspect.ratio = 1, strip.text = element_text(size = 6))
plot_threshold_pr
ggsave(file=paste0("results/figures/plot_threshold_pr", ".png"), plot=plot_threshold_pr, width=12, height=16)

# Figure: Precision-Recall AUC for all classes
plot_pr = ggplot(perf_present, aes(x = recall, y = precision, color = model)) +
  geom_path() +
  geom_path(aes(color = model)) +
  facet_wrap(~ label, ncol = 7, scales = "free_y") +
  scale_color_manual(values = c("Target" = "royalblue", "Source" = "salmon")) +
  labs(x = "Recall", y = "Precision") +
  theme_bw() + theme(aspect.ratio = 1)
plot_pr

# Figure 4 (1/2): Threshold performance for selected species classes
selected_species_prt = ggplot(perf_selected_species, aes(x = threshold)) +
  geom_path(aes(y = recall, linetype = "Recall", color = model)) +
  geom_path(aes(y = precision, linetype = "Precision", color = model)) +
  facet_wrap(~ label, ncol = 3) +
  scale_color_manual(values = c("Target" = "royalblue", "Source" = "salmon")) +
  scale_linetype_manual(values = c("Recall" = "dashed", "Precision" = "solid", "F1" = "dotted")) +
  labs(title = "Segment level test performance", x = "Threshold", y = "Performance", color = 'Model', linetype = 'Metric') +
  theme_bw() +
  theme(panel.grid.minor = element_blank(), aspect.ratio = 1)
selected_species_prt
ggsave(file=paste0("results/figures/selected_species_prt", ".png"), plot=selected_species_prt, width=8, height=6)

# Figure 4 (2/2): Confidence score distributions for selected species classes on the logit scale
plot_scores_selected_species = ggplot() +
  geom_density(data = subset(perf_selected_species, model == "Source"), aes(x = logit), color='salmon', fill='salmon', alpha=0.6) +
  geom_density(data = subset(perf_selected_species, model == "Target"), aes(x = logit), color='royalblue', fill='royalblue', alpha=0.6) +
  facet_wrap(~ label, ncol = 3) +
  scale_x_continuous(name = "Confidence (logit)", breaks = c(-15,0,15), limits = c(-15,15)) +
  ylim(0,0.3) +
  labs(x = "Confidence (logit)", y = "Density", title = "Prediction score distributions") +
  theme_bw() +
  theme(panel.grid.minor = element_blank())
plot_scores_selected_species
ggsave(file=paste0("results/figures/selected_species_plot_scores", ".svg"), plot=plot_scores_selected_species, width=6.81, height=3)

# Figure A.2: Confidence score distributions on the logit scale
plot_scores = ggplot() +
  geom_density(data = subset(perf, model == "Source"), aes(x = logit), color='salmon', fill='salmon', alpha=0.6) +
  geom_density(data = subset(perf, model == "Target"), aes(x = logit), color='royalblue', fill='royalblue', alpha=0.6) +
  facet_wrap(~ label, ncol = 8) +
  scale_x_continuous(name = "Confidence (logit)", breaks = c(-15,0,15), limits = c(-15,15)) +
  ylim(0,0.5) +
  labs(x = "Confidence (logit)", y = "Density", title = "Prediction score distributions") +
  theme_minimal() +
  theme(panel.grid.minor = element_blank())
plot_scores
ggsave(file=paste0("results/figures/plot_scores", ".png"), plot=plot_scores, width=14, height=12)

# Figure: Confidence score histogram on the 0,1 (sigmoid) scale
plot_histogram = ggplot(perf, aes(x = threshold)) +
  geom_histogram(data = subset(perf, model == "Source"), fill = "red", alpha = 0.55, bins = 12) +
  geom_histogram(data = subset(perf, model == "Target"), fill = "blue", alpha = 0.55, bins = 12) +
  facet_wrap(~ label, scales = "free_y") +
  scale_y_continuous(breaks = scales::pretty_breaks(n = 5)) +
  coord_cartesian(ylim = c(0, 40)) +
  labs(x = "Score Threshold", y = "Number of Detections") +
  theme_minimal()
plot_histogram
