# Global helpers for figure generation

labels_to_remove = c(
  # excluded due to bias
  "northern goshawk", "american goldfinch", "macgillivray's warbler", "american crow", "red-tailed hawk", "yellow warbler", "sharp-shinned hawk", "bald eagle",
  # non-relevant species  
  "long-eared owl", "american three-toed woodpecker", "cassin's vireo", "western kingbird", "eastern kingbird", "dusky flycatcher", "mountain bluebird", "vesper sparrow", "american redstart", "cassin's finch", "clark's nutcracker", "pine grosbeak", "lazuli bunting", "bushtit", "ring-necked pheasant", "california quail"
)

site_presence_absence = read.csv('data/test/site_presence_absence.csv')

label_counts = data.frame()
for (label in site_presence_absence[3:nrow(site_presence_absence),1]) {
  print(label)
  count = sum(na.omit(as.numeric(site_presence_absence[site_presence_absence[,1] == label, 2:ncol(site_presence_absence)])))
  print(count)
  label_counts = rbind(label_counts, data.frame(
    label = label,
    count = count
  ))
}
