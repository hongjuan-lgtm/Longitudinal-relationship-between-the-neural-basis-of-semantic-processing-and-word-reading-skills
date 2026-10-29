library(ggplot2)
library(patchwork)
library(dplyr)

# Load data
data_file <- "/path/to/regression_refinement_deep_7-9_AG.RData"

env <- new.env()
load(data_file, envir = env)
df <- env$df2

required_cols <- c("WordReading", "IQ", "beta_T1", "beta_T2",
                   "beta_L_T1", "beta_L_T2", "beta_C_T1", "beta_C_T2")

df <- df[complete.cases(df[, required_cols]), , drop = FALSE]

# Standardized residuals: control for baseline activation and IQ
df$Deep_res <- as.numeric(scale(resid(lm(beta_T2 ~ beta_T1 + IQ, data = df))))
df$SL_res <- as.numeric(scale(resid(lm(beta_L_T2 ~ beta_L_T1 + IQ, data = df))))
df$SC_res <- as.numeric(scale(resid(lm(beta_C_T2 ~ beta_C_T1 + IQ, data = df))))

plot_data <- bind_rows(
  df %>% transmute(WordReading, Condition = "SL", Residual = SL_res),
  df %>% transmute(WordReading, Condition = "SC", Residual = SC_res)
) %>%
  mutate(Condition = factor(Condition, levels = c("SL", "SC")))

# R² and p from residual regressions
get_stats <- function(data, formula) {
  s <- summary(lm(formula, data = data))
  data.frame(R2 = s$r.squared, p = s$coefficients["WordReading", "Pr(>|t|)"])
}

deep_stats <- get_stats(df, Deep_res ~ WordReading)

condition_stats <- plot_data %>%
  group_by(Condition) %>%
  group_modify(~ get_stats(.x, Residual ~ WordReading)) %>%
  ungroup()

print(deep_stats)
print(condition_stats)

# Format annotations
format_stat <- function(x) {
  ifelse(x < 0.001, "< 0.001", sprintf("= %.3f", x))
}

make_label <- function(name, r2, p) {
  paste0(name, ": R² ", format_stat(r2), ", p ", format_stat(p))
}

label_deep <- make_label("Deep (LowAssoc - Perc)", deep_stats$R2, deep_stats$p)

condition_stats <- condition_stats %>%
  mutate(
    description = if_else(Condition == "SL", "LowAssoc", "Perc"),
    label = make_label(description, R2, p)
  )

label_L <- condition_stats$label[condition_stats$Condition == "SL"]
label_C <- condition_stats$label[condition_stats$Condition == "SC"]

# Shared axes
padded_range <- function(x) {
  limits <- range(x, na.rm = TRUE)
  limits + c(-1, 1) * diff(limits) * 0.05
}

x_limits <- padded_range(df$WordReading)
y_limits <- padded_range(c(df$Deep_res, plot_data$Residual))

common_axes <- list(
  scale_x_continuous(limits = x_limits, breaks = seq(10, 70, 10)),
  scale_y_continuous(limits = y_limits, breaks = seq(-3, 4, 1)),
  labs(x = "Word Reading at T1", y = "Scaled Residual of AG Activation at T2"),
  theme_classic(base_size = 14),
  theme(axis.title.x = element_text(size = 20), axis.title.y = element_text(size = 18))
)

# Panel A: deep contrast
p_left <- ggplot(df, aes(WordReading, Deep_res)) +
  geom_point(color = "black", size = 2.5, alpha = 0.7, shape = 8) +
  geom_smooth(method = "lm", formula = y ~ x, se = TRUE,
              color = "black", fill = "grey70", linewidth = 1, alpha = 0.4) +
  annotate("text", x = Inf, y = Inf, label = label_deep,
           hjust = 1.1, vjust = 1.5, size = 5) +
  common_axes

# Panel B: separate Low association and Control conditions
p_right <- ggplot(plot_data, aes(WordReading, Residual, color = Condition, shape = Condition)) +
  geom_point(alpha = 0.6, size = 2.5) +
  geom_smooth(aes(fill = Condition), method = "lm", formula = y ~ x, se = TRUE, linewidth = 1) +
  scale_color_manual(values = c(SL = "grey30", SC = "grey50")) +
  scale_fill_manual(values = c(SL = "grey30", SC = "grey70")) +
  scale_shape_manual(values = c(SL = 8, SC = 5)) +
  annotate("text", x = Inf, y = Inf, label = label_L,
           hjust = 1.05, vjust = 1.5, color = "grey30", size = 5) +
  annotate("text", x = Inf, y = Inf, label = label_C,
           hjust = 1.05, vjust = 3, color = "grey50", size = 5) +
  common_axes +
  theme(legend.position = "none")

# Combine and save
combined_plot <- ((p_left | p_right) + plot_annotation(tag_levels = "A")) &
  theme(
    plot.tag = element_text(face = "bold", size = 20, hjust = 0, vjust = 1),
    plot.tag.position = c(-0.1, 1.1),
    plot.margin = margin(t = 20, r = 10, b = 10, l = 25)
  )

print(combined_plot)
ggsave("AG_combined_figure.png", plot = combined_plot, width = 12, height = 5, dpi = 300)
