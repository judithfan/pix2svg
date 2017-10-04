#!/usr/bin/env Rscript
knitr::opts_chunk$set(echo = TRUE)
library(tidyverse)
library(ggthemes)
library(lme4)
library(jsonlite)
library(stringr)
library(optparse)


# Handle command line options
option_list = list(
  make_option(c("-j", "--json"), type="character", default=NULL, 
              help="json file path", metavar="character"),
  make_option(c("-o", "--out"), type="character", default="./",
              help="pdf plot directory", metavar="character")
); 
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

# build data from CSV file
raw <- read_delim('./data/sketchpad_basic_merged_group_data.csv', delim = ',') 
d <- raw %>% 
  mutate(gameid = str_sub(gsub("-", "", gameID), start = -12)) %>%
  mutate(sketchLabel = sprintf('%s_%s', gameid, trialNum)) %>%
  filter(iteration == 'pilot1') %>% # Just use pilot 1 for now
  select(sketchLabel, condition, target, Distractor1, Distractor2, Distractor3, mean_intensity, pose) 
d$context = apply(d %>% select(Distractor1, Distractor2, Distractor3), 1, 
                  function(x) paste(sort(x), collapse="_"))
d.ordered = d %>% 
  select(-Distractor1, -Distractor2, -Distractor3) %>%
  separate(context, into = c('Distractor1', 'Distractor2', 'Distractor3')) %>%
  mutate(Target = sprintf("%s_%04d", target, pose)) %>%
  mutate(Distractor1 = sprintf("%s_%04d", Distractor1, pose)) %>%
  mutate(Distractor2 = sprintf("%s_%04d", Distractor2, pose)) %>%
  mutate(Distractor3 = sprintf("%s_%04d", Distractor3, pose))

similarities <- fromJSON(opt$json, flatten = T)
lookupSimilarity <- function(object, sketch) {
  return(similarities[[object]][[sketch]])
}

d.similarity <- d.ordered %>%
  gather(contextElement, name, Target, Distractor1, Distractor2, Distractor3) %>%
  rowwise() %>%
  mutate(similarity = lookupSimilarity(name, sketchLabel))

# plot density when comparing FAR conditions
tmp = d.similarity %>% 
  mutate(contextElement = case_when(contextElement == 'Target' ~ 'target',
                                    T ~ 'distractor')) %>%
  filter(condition == 'further') 

tmp %>%
  ggplot(aes(x = similarity, fill = contextElement)) +
    geom_density(alpha = .5) +
    theme_few() +
    theme(aspect.ratio = 1) +
    xlim(-1, 1) +
    ggtitle('sketch similarity in FAR condition') +
    scale_fill_colorblind()

ggsave(file.path(opt$out, 'far.pdf'))

# plot density when comparing CLOSE conditions
tmp = d.similarity %>% 
  mutate(contextElement = case_when(contextElement == 'Target' ~ 'target',
                                    T ~ 'distractor')) %>%
  filter(condition == 'closer') 

tmp %>%
  ggplot(aes(x = similarity, fill = contextElement)) +
    geom_density(alpha = .5) +
    theme_few() +
    theme(aspect.ratio = 1) +
    xlim(-1, 1) +
    ggtitle('sketch similarity in CLOSE condition') +
    scale_fill_colorblind()

ggsave(file.path(opt$out, 'close.pdf'))
