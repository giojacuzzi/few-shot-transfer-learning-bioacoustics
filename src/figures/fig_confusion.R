#############################################
# Visualize relationships between labels within a confusion matrix of incorrectly labeled species predictions
# via hierarchical edge bundling, organizing labels by type, order, and family. This identifies recurrent
# sources of confusion and novel signals that define the unique acoustic context of the target domain. 
#
# Adapted from https://r-graph-gallery.com/309-intro-to-hierarchical-edge-bundling.html
#
# Input:
# - Confusion matrix of incorrect prediction labels
# - Table of figure labels and group membership (e.g. order, family)
# - Complete segment level performance metrics for the model under consideration
#
# Output:
# - Hierarchical edge bundling plot (Fig 3)
#
# User-defined parameters
model_stub = 'OESF_1.0'
model_to_evaluate = 'source'
#############################################

library(ggraph)
library(igraph)
library(dplyr)
library(cowplot)
library(scales)
library(dplyr)
library(reshape2)
library(ggraph)
library(igraph)
library(tidyverse)
library(RColorBrewer)
source('src/figures/global.R')

node_alpha_max = 0.2

# Load requisite data
path_confusion_mtx = paste('results/', model_stub, '/test/segment_perf/', model_to_evaluate, '/confusion_matrix/confusion_matrix_T0.5.csv', sep='')
confusion_mtx = read.csv(path_confusion_mtx, row.names=1, check.names = FALSE)

path_perf_metrics = paste('results/', model_stub, '/test/segment_perf/metrics_complete.csv', sep='')
perf_metrics = read.csv(path_perf_metrics)

labels = read.csv('data/figures/fig_labels.csv')
perf_metrics = perf_metrics[perf_metrics$model == model_to_evaluate,]
perf_metrics$label = tolower(perf_metrics$label)
perf_metrics$PR_AUC[is.na(perf_metrics$PR_AUC)] = 0.5 # override missing AUC values for absent classes for visibility

confusion_mtx = confusion_mtx[, !names(confusion_mtx) %in% labels_to_remove]
confusion_mtx = confusion_mtx[!rownames(confusion_mtx) %in% labels_to_remove, ]

# Organize hierarchy
get_group = function(x) {
  if (grepl("biotic", x)) {
    strsplit(x, " ")[[1]][1]
  } else if (grepl("artifact", x)) {
    "artifact"
  } else if (grepl("other ", x)) {
    "abiotic"
  } else if (x == "origin") {
    NA
  } else {
    "biotic (source)"
  }
}

get_order = function(x) {
  if (grepl("other ", x)) {
    y = "Abiotic"
  } else {
    y = labels[labels['label'] == x, 'order'][1]
  }
  return(y)
}

get_family = function(x) {
  if (grepl("other ", x)) {
    y = "AaOther"
  } else{
    y = labels[labels['label'] == x, 'family'][1] 
  }
  return(y)
}

get_presence = function(x) {
  p = ifelse(perf_metrics[perf_metrics$label == x, 'N_pos'] > 0, 1.0, 0.0)
  if (length(p) == 0) {
    return(0)
  }
  return(p)
}

groups = sapply(rownames(confusion_mtx), get_order)
groups = ifelse(is.na(groups), "Other", groups)

origin_to_groups = data.frame(from="origin", to=unique(groups))
groups_to_subgroups = data.frame(from = unname(groups), to = names(groups))
groups_to_subgroups$family = sapply(groups_to_subgroups$to, get_family)
groups_to_subgroups = groups_to_subgroups %>% arrange(from, family, to) %>% select(-family)
priority_levels = c("Artifact", "Abiotic", "Biotic")
groups_to_subgroups$priority = ifelse(groups_to_subgroups$from %in% priority_levels, match(groups_to_subgroups$from, priority_levels), length(priority_levels) + 1)
groups_to_subgroups = groups_to_subgroups[order(groups_to_subgroups$priority), ]
groups_to_subgroups = groups_to_subgroups %>% select(-priority)

hierarchy = rbind(origin_to_groups, groups_to_subgroups)
edges = rbind(origin_to_groups, groups_to_subgroups)

# One line per object of the hierarchy, giving features of nodes.
vertices = data.frame(name = unique(c(as.character(hierarchy$from), as.character(hierarchy$to))) )
vertices$Group = sapply(vertices$name, get_order)
vertices$Presence = as.numeric(sapply(vertices$name, get_presence))
vertices[vertices$Group %in% c('Abiotic', 'Biotic'), 'Presence'] = 1

# Retrieve PR AUC
vertices = vertices %>%
  left_join(perf_metrics %>% select(label, PR_AUC), by = c("name" = "label")) %>%
  rename(AUC = PR_AUC) %>%
  mutate(
    Presence = ifelse(is.na(Presence), 1.0, Presence),
    AUC = ifelse(is.na(AUC), 1.0, AUC)
  )

# Create a graph network object and visualize to validate relationships
graph_network = graph_from_data_frame(hierarchy, vertices=vertices)
plot(graph_network, vertex.label="", edge.arrow.size=0, vertex.size=2)
ggraph(graph_network, layout = 'dendrogram', circular = FALSE) + 
  geom_edge_link() +
  theme_void()
ggraph(graph_network, layout = 'dendrogram', circular = TRUE) + 
  geom_edge_diagonal() +
  theme_void()
from = rownames(confusion_mtx)
to = colnames(confusion_mtx)

# Establish connections
connections = melt(as.matrix(confusion_mtx), varnames = c("from", "to"), value.name = "value")
connections = connections[connections$value > 0, ]
connections = connections[connections$from != connections$to, ]

# Add additional calculations for visualization
vertices$id = NA
leaves = which(is.na( match(vertices$name, edges$from) ))
n_leaves = length(leaves)
vertices$id[ leaves ] = seq(1:n_leaves)
vertices$angle = 90.0 - 360.0 * vertices$id / n_leaves
vertices$hjust = ifelse(vertices$angle < -90.0, 1, 0)
vertices$angle = ifelse(vertices$angle < -90.0, vertices$angle+180.0, vertices$angle)

graph_network = igraph::graph_from_data_frame( edges, vertices=vertices )

# Refer connection object leaf ids
connect_from = match( connections$from, vertices$name)
connect_to =match( connections$to, vertices$name)

# Plot hierarchical edge bundling
g = ggraph(graph_network, layout = 'dendrogram', circular = TRUE) + 
  geom_conn_bundle(data = get_con(from = connect_from, to = connect_to), alpha=0.35, width=0.5, aes(colour=..index..), tension = 0.98) +
  scale_edge_colour_distiller(palette = "RdPu") +
  geom_node_text(aes(x = x*1.12, y=y*1.12, filter = leaf, label=tools::toTitleCase(gsub("biotic |abiotic |other ", "", name)), angle = angle, hjust=hjust, colour=Group), size=2, alpha=1) +
  
  geom_node_point(aes(filter = leaf, x = x*1.07, y=y*1.07, colour=Group, size=AUC, alpha=Presence)) +
  scale_alpha(range=c(0.2,0.9)) + # for nodes
  scale_color_manual(values=c("#777777", "#F8766D", "#DE8C00", "#222222", "#00BA38", "#00C08B", "#00BFC4", "#00B4F0", "#619CFF", "#C77CFF", "#F564E3", "#FF64B0")) +
  scale_size_continuous(range = c(4.75,0.25)) +
  theme_void() +
  expand_limits(x = c(-1.75, 1.75), y = c(-1.75, 1.75))

# Plot legend
plot(g + theme(
  legend.position="none",
))
legend = cowplot::get_legend(g)
plot_grid(legend, ncol = 1)

ggsave(file="results/figures/confusion.svg", plot=g + theme(legend.position="none"), width=8, height=8)
ggsave(file="results/figures/confusion_legend.svg", plot=legend, width=8, height=8)
