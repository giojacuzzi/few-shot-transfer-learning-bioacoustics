# Calculate and plot results of training sample size experiment
#
# Input:
# - Segment level performance metrics for target models in the sample size experiment at TODO "results/{target_model_stub}/test/sample_perf/threshold_perf_{model_tag}"
# -  Sample size counts for target models in the sample size experiment at TODO ""
#
# Output:
# - Plots for training sample size performance experiment (Fig A.3)

library(dplyr)
library(ggplot2)
library(viridis)
library(stringr)

# Retrieve all segment level performance metrics for target models in the sample size experiment
files_metrics = list.files(path = 'data/results', pattern = "\\metrics_custom.csv$", full.names = TRUE, recursive = TRUE)
metrics = data.frame()
for (file in files_metrics) {
  data = read.csv(file)
  data$label = tolower(data$label)
  metrics = rbind(metrics, data)
}
metrics$model_dir = gsub("data/test/|/custom", "", metrics$model)

# Count the number of training samples for each class
files_class_counts = unique(file.path('data/models/target', paste0(metrics$model_dir, '/trained_class_counts.csv')))
class_counts = data.frame()
for (file in files_class_counts) {
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
if (FALSE) {
  for (class_label in unique(data$label)) {
    data_l = data[data$label == class_label,]
    print(data_l[, c('label', 'PR_AUC', 'count')] %>% arrange(count))
    p = ggplot() +
      geom_line(data = data_l, aes(x = count, y = PR_AUC)) +
      geom_point(data = data_l, aes(x = count, y = PR_AUC)) +
      labs(x = "Training samples", y = "PR AUC", title = class_label) +
      theme_minimal(); print(p)
    readline()
  }
}

# Figure A.3: Plot relationships between training sample size and precision-recall AUC performance across all species classes
plot_sample_size = ggplot(data = data) +
  geom_line(aes(x = count, y = PR_AUC, color = count)) +
  scale_colour_viridis(option = "D") +
  facet_wrap(~ str_to_title(label), ncol = 7, scales = "free_x") +
  theme_bw() + theme(aspect.ratio = 1) +
  labs(color = "Samples", x = "Training sample size", y = "PR AUC")
plot_sample_size
ggsave(file=paste0("data/figures/plot_sample_size", ".png"), plot=plot_sample_size, width=12, height=16)

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
