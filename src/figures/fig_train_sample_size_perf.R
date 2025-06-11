##########################################################################################
# Calculate and plot results of training sample size experiment
#
# Input:
# - Segment level performance metrics for target models in the sample size experiment (e.g. "results/{model}/test/segment_perf/metrics_target.csv")
# - Sample size counts for target models in the sample size experiment (e.g. "data/cache/{model_root}/training/{model}/trained_class_counts.csv")
#
# Output:
# - Plots for training sample size performance experiment (Fig A.3)
#
# User-defined parameters
models = c(
  'target_S1_LR0.001_BS10_HU0_LSFalse_US0_N1_I1',
  'target_S1_LR0.001_BS10_HU0_LSFalse_US0_N5_I1',
  'target_S1_LR0.001_BS10_HU0_LSFalse_US0_N10_I1',
  'target_S1_LR0.001_BS10_HU0_LSFalse_US0_N50_I1',
  'target_S1_LR0.001_BS10_HU0_LSFalse_US0_N100_I1',
  'target_S1_LR0.001_BS10_HU0_LSFalse_US0_N125_I0' # NOTE: This model is the final OESF_1.0 model
)
##########################################################################################

library(dplyr)
library(ggplot2)
library(viridis)
library(stringr)

model_paths = data.frame(
  metrics      = paste0('results/', models, '/test/segment_perf/metrics_target.csv'),
  class_counts = paste0('data/cache/target_S1_LR0.001_BS10_HU0_LSFalse_US0/training/', models, '/trained_class_counts.csv')
)

# Retrieve all segment level performance metrics for target models in the sample size experiment
metrics = data.frame()
for (file in model_paths$metrics) {
  data = read.csv(file)
  data$label = tolower(data$label)
  data$model_dir = strsplit(file, '/')[[1]][2]
  metrics = rbind(metrics, data)
}

# Count the number of training samples for each class
class_counts = data.frame()
for (file in model_paths$class_counts) {
  data = read.csv(file)
  names(data)[names(data) == "labels"] <- "label"
  data$label = tolower(sub(".*_", "", data$label))
  data$model_dir = basename(dirname(file))
  class_counts = rbind(class_counts, data)
}

# Merge training sample counts with performance metrics
data = merge(metrics, class_counts, by = c("label", "model_dir"), all.x = TRUE)

# Include origin point (0.0 PR AUC at 0 training samples)
origin_points = data.frame(label = unique(data$label))
origin_points$PR_AUC = 0.0
origin_points$count = 0
data = bind_rows(data, origin_points)

# Inspect performance for individual classes
class_labels = c('pileated woodpecker', 'pacific wren', 'sooty grouse')
for (class_label in class_labels) {
  data_l = data[data$label == class_label,]
  print(data_l[, c('label', 'PR_AUC', 'count')] %>% arrange(count))
  p = ggplot() +
    geom_line(data = data_l, aes(x = count, y = PR_AUC)) +
    scale_x_continuous(breaks = seq(0, 150, by = 10)) +
    scale_y_continuous(breaks = seq(0, 1.0, by = 0.1)) +
    geom_point(data = data_l, aes(x = count, y = PR_AUC)) +
    labs(x = "Training samples", y = "PR AUC", title = class_label) +
    theme_minimal(); print(p)
}

# Figure A.3: Plot relationships between training sample size and precision-recall AUC performance across all species classes
plot_sample_size = ggplot(data = data) +
  geom_line(aes(x = count, y = PR_AUC, color = count), linewidth = 0.75, lineend = 'round') +
  scale_colour_viridis_c(option = "D", limits = c(0,150), breaks = c(0, 25, 50, 75, 100, 125, 150)) +
  facet_wrap(~ str_trunc(str_to_title(label), width=24), ncol = 6) +
  scale_x_continuous(breaks = seq(0, 125, by = 25)) +  # Only set breaks, no limits
  coord_cartesian(xlim = c(0, 130)) +  # Zoom without dropping data
  labs(color = "Samples", x = "Training sample size", y = "PR AUC") +
  theme_bw() + theme(aspect.ratio = 1, strip.text = element_text(size = 6))
plot_sample_size
ggsave(file=paste0("results/figures/plot_sample_size", ".png"), plot=plot_sample_size, width=12, height=16)

# Calculate the mean PR AUC for each training sample size for species classes
labels_to_exclude = c('abiotic aircraft', 'abiotic logging', 'abiotic rain', 'abiotic vehicle', 'abiotic wind', 'background', 'biotic anuran', 'biotic insect')
mean_sample_size = class_counts[!class_counts$label %in% labels_to_exclude,] %>% group_by(model_dir) %>% summarise(mean_sample_size = mean(count)) %>% ungroup()
mean_PR_AUC = data[!data$label %in% labels_to_exclude,] %>% group_by(model_dir) %>% summarise(mean_PR_AUC = mean(PR_AUC)) %>% ungroup()
mean_data = full_join(mean_sample_size, mean_PR_AUC, by = 'model_dir') %>% arrange(mean_sample_size); mean_data
ggplot(mean_data, mapping = aes(x = mean_sample_size, y = mean_PR_AUC)) +
  geom_point() +
  geom_line() +
  geom_hline(yintercept = 0.79, linetype = "dashed", color = "red") + # source model mean PR AUC
  theme_bw()
