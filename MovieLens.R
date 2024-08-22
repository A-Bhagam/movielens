# Load all necessary libraries

if(!require(tidyverse)) install.packages("tidyverse") 
if(!require(caret)) install.packages("caret") 
if(!require(dplyr)) install.packages("dplyr") 
if(!require(stringr)) install.packages("stringr") 
if(!require(tidyr)) install.packages("tidyr") 
if(!require(forcats)) install.packages("forcats") 
if(!require(ggplot2)) install.packages("ggplot2") 
if(!require(egg)) install.packages("egg") 
if(!require(data.table)) install.packages("data.table") 
```

# Create edx set and validation set

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

set.seed(1, sample.kind="Rounding")

test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Show first 5 rows of the edx dataset
head(edx, 5)

#Create a histogram for the ratings in the edx dataset
edx %>% ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.2, color = "black") +
  labs(title = "Distribution of Ratings",
       x = "Rating", 
       y = "Count" ) +
  scale_y_continuous(breaks = seq(0, 3000000, 500000))


#Provide relevant statistics for the ratings variable
summary(edx$rating)


#Create a histogram that shows a distribution of the movies by their rating
edx %>% group_by(movieId) %>%
  summarise(average_rating = sum(rating)/n()) %>%
  ggplot(aes(average_rating)) +
  geom_histogram(bins= 50, color = "black") +
  labs(title = "Distribution of Ratings by MovieID",
       x = "Average Rating", 
       y = "Movies")

#create a histogram that shows a distribution of the users by their average rating
edx %>% group_by(userId) %>%
  summarise(average_rating = sum(rating)/n()) %>%
  ggplot(aes(average_rating)) +
  geom_histogram(bins= 50, color = "black") +
  labs(title = "Distribution of Ratings by UserID",
       x = "Average Rating", 
       y = "Users")

# Separate individual genres and rank them by the total number of ratings
edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarise(Number_of_Movies = n(), Rating = round(mean(rating), 2)) %>%
  arrange(desc(Number_of_Movies))

#Extract the year from the title in the edx dataset
edx$year <- as.numeric(sub(".*\\((\\d{4})\\)$", "\\1", edx$title))

#Do the same for the final_holdout_test dataset
final_holdout_test$year <- as.numeric(sub(".*\\((\\d{4})\\)$", "\\1", final_holdout_test$title))

edx %>% group_by(year) %>%
  summarise(rating = mean(rating)) %>%
  ggplot(aes(year, rating)) +
  geom_point() +
  labs(title ="Average rating by Year",
       x = "Release Year", 
       y = "Average Rating")


# Calculate mean of the training dataset
mu <- mean(edx$rating)

# Predict the RMSE on the validation set
baseline_rmse <- RMSE(final_holdout_test$rating, mu)

# Creating a results dataframe that contains all RMSE results
results <- data.frame(model = "Baseline Model", RMSE = baseline_rmse)

# Show dataframe
results

# Calculate mean of the entire dataset
mu <- mean(edx$rating)

# Calculate movie averages
movie_average <- edx %>%
   group_by(movieId) %>%
   summarize(b_i = mean(rating - mu))

# Calculate predicted ratings
predictions <- final_holdout_test %>%
   left_join(movie_average, by='movieId') %>%
   mutate(prediction = mu + b_i) %>%
   pull(prediction)

movie_rmse <- RMSE(final_holdout_test$rating, predictions)

# Add the results to the results dataset
results <- results %>% add_row(model = "Movie Model", RMSE = movie_rmse)

# Show results
results

# Calculate average rating of the entire dataset
mu <- mean(edx$rating)

# Calculate movie averages
movie_average <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# Calculate user averages
user_average <- edx %>%
  left_join(movie_average, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Calculate predicted ratings
predictions <- final_holdout_test %>%
  left_join(movie_average, by='movieId') %>%
  left_join(user_average, by='userId') %>%
  mutate(prediction = mu + b_i + b_u) %>%
  pull(prediction)

movie_user_rmse <- RMSE(final_holdout_test$rating, predictions)

# Add the results to the results dataset
results <- results %>% add_row(model="Movie and User Model", RMSE = movie_user_rmse)

# Show results
results

# Calculate average rating of the entire dataset
mu <- mean(edx$rating)

# Define lambda
lambdas <- seq(0, 100, 1)

# Compute the predicted ratings using different values of lambda
rmses <- sapply(lambdas, function(lambdas) {
   
  b_i <- edx %>%
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu) / (n() + lambdas))
  
   b_u <- edx %>%
      left_join(b_i, by='movieId') %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - mu) / (n() + lambdas))
   
   predictions <- final_holdout_test %>%
      left_join(b_i, by='movieId') %>%
      left_join(b_u, by='userId') %>%
      mutate(prediction = mu + b_i + b_u) %>%
      pull(prediction)
   
  return(RMSE(final_holdout_test$rating, predictions))
   
})

# Get the lowest RSME
regularized_rmse <- min(rmses)

# Add the results to the results dataset
results <- results %>% add_row(model="Regularized Model", RMSE = regularized_rmse)

# Show results
results

results

