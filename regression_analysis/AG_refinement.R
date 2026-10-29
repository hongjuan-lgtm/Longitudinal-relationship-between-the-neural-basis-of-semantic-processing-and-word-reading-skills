library(ggplot2)
library(dplyr)
library(lmerTest)

# Load data
deep_file <- "/path/to/regression_refinement_deep_7-9_AG_withbehavior.RData"
shallow_file <- "/path/to/regression_refinement_shallow_7-9_AG.RData"

load_df <- function(file) {
  env <- new.env()
  load(file, envir = env)
  if (!exists("df2", envir = env, inherits = FALSE)) stop("df2 not found in: ", file)
  env$df2
}

deep_df <- load_df(deep_file)
shallow_df <- load_df(shallow_file)

# Match participants and keep complete cases
df_clean <- deep_df %>%
  select(participant_id, deep_T1 = AG_T1, deep_T2 = AG_T2, WordReading, IQ) %>%
  left_join(
    shallow_df %>% select(participant_id, shallow_T1 = beta_T1, shallow_T2 = beta_T2),
    by = "participant_id"
  ) %>%
  mutate(participant_id = factor(participant_id)) %>%
  filter(if_all(-participant_id, ~ !is.na(.x)))

# Convert to long format
df_long <- bind_rows(
  df_clean %>% transmute(participant_id, Condition = "shallow", AG_T1 = shallow_T1, AG_T2 = shallow_T2, WordReading, IQ),
  df_clean %>% transmute(participant_id, Condition = "deep", AG_T1 = deep_T1, AG_T2 = deep_T2, WordReading, IQ)
) %>%
  mutate(Condition = factor(Condition, levels = c("shallow", "deep")))

# Mixed-effects model
model_interaction <- lmerTest::lmer(
  AG_T2 ~ Condition * (WordReading + AG_T1 + IQ) + (1 | participant_id),
  data = df_long, REML = FALSE
)

summary(model_interaction)

# Scale continuous-predictor slopes and their interactions
B <- lme4::fixef(model_interaction)
beta_standardized <- B
sd_y <- sd(df_long$AG_T2)

for (variable in c("WordReading", "AG_T1", "IQ")) {
  terms <- c(variable, paste0("Conditiondeep:", variable))
  beta_standardized[terms] <- B[terms] * sd(df_long[[variable]]) / sd_y
}

beta_standardized

# Calculate standardized residuals separately for each condition
df_plot <- df_long %>%
  group_by(Condition) %>%
  group_modify(~ {
    model <- lm(AG_T2 ~ AG_T1 + IQ, data = .x)
    mutate(.x, Residual = as.numeric(scale(resid(model))))
  }) %>%
  ungroup()

# Descriptive regressions for plot annotations
stats_data <- df_plot %>%
  group_by(Condition) %>%
  group_modify(~ {
    model <- lm(Residual ~ WordReading, data = .x)
    stats <- summary(model)
    data.frame(R2 = stats$r.squared, p = stats$coefficients["WordReading", "Pr(>|t|)"])
  }) %>%
  ungroup()

print(stats_data)

# Format plot labels
format_stat <- function(x) {
  ifelse(x < 0.001, "< 0.001", sprintf("= %.3f", x))
}

stats_data <- stats_data %>%
  mutate(
    description = if_else(Condition == "deep", "Deep (LowAssoc - Perc)", "Shallow (HighAssoc - Perc)"),
    label = paste0(description, ": R² ", format_stat(R2), ", p ", format_stat(p))
  )

label_deep <- stats_data$label[stats_data$Condition == "deep"]
label_shallow <- stats_data$label[stats_data$Condition == "shallow"]

# Plot
condition_colors <- c(shallow = "grey70", deep = "grey30")
condition_shapes <- c(shallow = 16, deep = 8)

p_interaction <- ggplot(df_plot, aes(WordReading, Residual, color = Condition, shape = Condition)) +
  geom_point(alpha = 0.6, size = 2.5) +
  geom_smooth(method = "lm", se = TRUE, linewidth = 1) +
  scale_color_manual(values = condition_colors, breaks = c("shallow", "deep")) +
  scale_shape_manual(values = condition_shapes, breaks = c("shallow", "deep")) +
  labs(x = "Word Reading at T1", y = "Scaled Residual of AG Activation at T2") +
  annotate("text", x = Inf, y = Inf, label = label_deep,
           hjust = 1.05, vjust = -0.5, color = "grey30", size = 9) +
  annotate("text", x = Inf, y = Inf, label = label_shallow,
           hjust = 1.05, vjust = -2, color = "grey60", size = 9) +
  coord_cartesian(clip = "off") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "none",
    axis.title = element_text(size = 24),
    axis.text = element_text(size = 20),
    plot.margin = margin(t = 80, r = 15, b = 15, l = 15)
  )

print(p_interaction)
ggsave("AG_interaction.png", plot = p_interaction, width = 12, height = 10, dpi = 300)
