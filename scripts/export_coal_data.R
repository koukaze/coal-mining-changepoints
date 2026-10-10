# Export chronological intervals from the pinned public boot dataset.

if (!requireNamespace("boot", quietly = TRUE)) {
    stop("Install boot version 1.3-32 before running this optional export.")
}
if (as.character(utils::packageVersion("boot")) != "1.3-32") {
    stop("This export requires boot version 1.3-32.")
}

args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args) > 0L) args[[1L]] else "data/coal_intervals_days.csv"
intervals <- as.integer(round(diff(boot::coal$date) * 365.25))
stopifnot(length(intervals) == 190L, sum(intervals) == 40549L)
stopifnot(all(intervals >= 0L), sum(intervals == 0L) == 1L)

dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
connection <- file(output, open = "wb")
writeLines(c("interval_days", as.character(intervals)), connection, sep = "\n")
close(connection)
message("Wrote ", length(intervals), " intervals to ", output)
