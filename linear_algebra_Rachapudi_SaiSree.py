""" LINEAR ALGEBRA FOR MACHINE LEARNING """

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

print("=" * 60)
print("SESSION 1: LINEAR ALGEBRA - FROM MATH TO ML")
print("=" * 60)

# ==============================================================================
# PART 1: VECTORS - THE FOUNDATION
# ==============================================================================

print("\n[PART 1] VECTORS: Your First ML Data Structure\n")

# Real-world example: User movie ratings
print("Example: Movie Recommendation System")
print("-" * 40)

# User ratings for 5 movies (scale 1-5)
user1_ratings = np.array([5, 3, 0, 4, 2])
user2_ratings = np.array([4, 0, 0, 5, 1])
user3_ratings = np.array([1, 4, 5, 2, 5])
user4_ratings = np.array([3, 2, 1, 1, 4])
movies = ['Action', 'Comedy', 'Horror', 'Sci-Fi', 'Romance']

print("User 1 ratings:", dict(zip(movies, user1_ratings)))
print("User 2 ratings:", dict(zip(movies, user2_ratings)))
print("User 3 ratings:", dict(zip(movies, user3_ratings)))
print("User 4 ratings:", dict(zip(movies, user4_ratings)))
# Vector operations
print("\nVector Operations:")
print("1. Addition (combining preferences):")
combined = user1_ratings + user2_ratings
print(f"   User1 + User2 = {combined}")

print("\n2. Scalar Multiplication (amplifying preferences):")
amplified = user1_ratings * 2
print(f"   User1 * 2 = {amplified}")

print("\n3. Magnitude (how strong are preferences?):")
magnitude = np.linalg.norm(user1_ratings)
print(f"   ||User1|| = {magnitude:.2f}")

# STUDENT EXERCISE
print("\n" + "=" * 60)
print("EXERCISE 1: Create your own rating vector for 5 movies")
print("Calculate its magnitude and compare with User1")
print("=" * 60)

magnitude_user1 = np.linalg.norm(user1_ratings)
magnitude_user4 = np.linalg.norm(user4_ratings)

print(f"||User1|| = {magnitude_user1:.2f}")
print(f"||User4|| = {magnitude_user4:.2f}")

if magnitude_user4 > magnitude_user1:
    print("→ User 4 has stronger preferences than User 1.")
else:
    print("→ User 1 has stronger preferences than User 4.")

# ==============================================================================
# PART 2: DOT PRODUCT - THE SIMILARITY MEASURE
# ==============================================================================

print("\n\n[PART 2] DOT PRODUCT: Finding Similar Users\n")

def cosine_similarity(vec1, vec2):
    """
    Calculate similarity between two vectors
    Returns value between -1 (opposite) and 1 (identical)
    """
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    
    return dot_product / (magnitude1 * magnitude2)

# Calculate similarities
sim_1_2 = cosine_similarity(user1_ratings, user2_ratings)
sim_1_3 = cosine_similarity(user1_ratings, user3_ratings)
sim_2_3 = cosine_similarity(user2_ratings, user3_ratings)

print("User Similarities:")
print(f"User1 <-> User2: {sim_1_2:.3f}")
print(f"User1 <-> User3: {sim_1_3:.3f}")
print(f"User2 <-> User3: {sim_2_3:.3f}")

print("\nInterpretation:")
print("• Values close to 1 = Very similar taste")
print("• Values close to 0 = No similarity")
print("• Values close to -1 = Opposite taste")

# Recommendation based on similarity
print("\n--- Making Recommendations ---")
if sim_1_2 > sim_1_3:
    print("User1 is more similar to User2")
    print("Recommend User2's favorites to User1!")
else:
    print("User1 is more similar to User3")
    print("Recommend User3's favorites to User1!")

# STUDENT EXERCISE
print("\n" + "=" * 60)
print("EXERCISE 2: Add a 4th user and find who they're most similar to")
print("Hint: Use the cosine_similarity function above")
print("=" * 60)
print("\n[Exercise 2] Similarities for User4:")
sim_4_1 = cosine_similarity(user4_ratings, user1_ratings)
sim_4_2 = cosine_similarity(user4_ratings, user2_ratings)
sim_4_3 = cosine_similarity(user4_ratings, user3_ratings)

print(f"User4 <-> User1: {sim_4_1:.3f}")
print(f"User4 <-> User2: {sim_4_2:.3f}")
print(f"User4 <-> User3: {sim_4_3:.3f}")

similarities = {
    "User1": sim_4_1,
    "User2": sim_4_2,
    "User3": sim_4_3
}

most_similar = max(similarities, key=similarities.get)
print(f"→ User 4 is most similar to: {most_similar}")
print(f"User4 is most similar to: {most_similar[0]}")
# ==============================================================================
# PART 3: MATRICES - ORGANIZING DATA
# ==============================================================================

print("\n\n[PART 3] MATRICES: Organizing Multiple Users\n")

# Create user-movie matrix
user_movie_matrix = np.array([
    [5, 3, 0, 4, 2],  # User 1
    [4, 0, 0, 5, 1],  # User 2
    [1, 4, 5, 2, 5],  # User 3
    [5, 3, 1, 4, 3],  # User 4
    [0, 5, 5, 1, 4],  # User 5
])

print("User-Movie Matrix Shape:", user_movie_matrix.shape)
print("(5 users × 5 movies)\n")
print(user_movie_matrix)

# Matrix operations
print("\n--- Matrix Operations ---")

# 1. Transpose (flip rows and columns)
print("\n1. Transpose (now it's Movies × Users):")
movie_user_matrix = user_movie_matrix.T
print("New shape:", movie_user_matrix.shape)

# 2. Mean ratings per movie
print("\n2. Average rating per movie:")
avg_ratings = np.mean(user_movie_matrix, axis=0)
for movie, rating in zip(movies, avg_ratings):
    print(f"   {movie}: {rating:.2f} ⭐")

# 3. Mean ratings per user  
print("\n3. Average rating per user:")
user_avgs = np.mean(user_movie_matrix, axis=1)
for i, avg in enumerate(user_avgs, 1):
    print(f"   User {i}: {avg:.2f} ⭐")

# STUDENT EXERCISE
print("\n" + "=" * 60)
print("EXERCISE 3: Find the most generous rater (highest average)")
print("Find the most critical rater (lowest average)")
print("=" * 60)
most_generous = np.argmax(user_avgs) + 1
most_critical = np.argmin(user_avgs) + 1
print("User averages:", user_avgs)
print(f"Most generous user → User {most_generous}")
print(f"Most critical user → User {most_critical}")

# ==============================================================================
# PART 4: MATRIX MULTIPLICATION - THE MAGIC
# ==============================================================================

print("\n\n[PART 4] MATRIX MULTIPLICATION: Where ML Happens\n")

# Simple neural network simulation
print("Example: A Simple Neural Network Layer")
print("-" * 40)

# Input: 3 features (height, weight, age)
input_data = np.array([170, 65, 25])  # [cm, kg, years]
print("Input (person's data):", input_data)

# Weights: how important is each feature?
# Let's predict: athletic ability
weights = np.array([
    [0.5, 0.3],   # height weights for 2 outputs
    [0.4, 0.2],   # weight weights
    [-0.1, 0.5]   # age weights
])
print("\nWeights shape:", weights.shape)

# Matrix multiplication
output = np.dot(input_data, weights)
print("\nOutput (predictions):", output)
print("This is how neural networks transform data!")

# Visualizing matrix multiplication
print("\n--- Image Transformation Example ---")

# Create a simple image (5x5 grid)
image = np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0]
])

print("Original 'Image':")
print(image)

# Rotation matrix (90 degrees)
def rotate_90(img):
    return np.rot90(img)

rotated = rotate_90(image)
print("\nRotated 90° (using matrix operations):")
print(rotated)

# STUDENT EXERCISE
print("\n" + "=" * 60)
print("EXERCISE 4: Create a 3x3 matrix and multiply it with weights")
print("Experiment with different values and observe outputs")
print("=" * 60)
matrix3 = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

result3 = matrix3.dot(weights)

print("3x3 matrix:\n", matrix3)
print("\nWeights:\n", weights)
print("\nResult (matrix3 × weights):\n", result3)
# ==============================================================================
# PART 5: PCA - DIMENSIONALITY REDUCTION
# ==============================================================================

print("\n\n[PART 5] PCA: Reducing Dimensions (The Mind-Bender)\n")

# Generate sample data: student scores
np.random.seed(42)
n_students = 100

# Correlated scores (math and physics are related)
math_scores = np.random.normal(75, 15, n_students)
physics_scores = math_scores + np.random.normal(0, 5, n_students)
chemistry_scores = np.random.normal(70, 20, n_students)
english_scores = np.random.normal(80, 10, n_students)

# Create dataset
student_data = np.column_stack([
    math_scores, 
    physics_scores, 
    chemistry_scores, 
    english_scores
])

print(f"Student Data: {n_students} students × 4 subjects")
print("Shape:", student_data.shape)
print("\nFirst 5 students:")
print(student_data[:5])

# Apply PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(student_data)

print("\nAfter PCA: 4 dimensions → 2 dimensions")
print("New shape:", reduced_data.shape)
print("\nVariance explained by each component:")
for i, var in enumerate(pca.explained_variance_ratio_, 1):
    print(f"Component {i}: {var*100:.1f}%")

print(f"\nTotal variance retained: {sum(pca.explained_variance_ratio_)*100:.1f}%")

# Visualization code (students can run this)
print("\n" + "=" * 60)
print("VISUALIZATION CODE (Run this separately):")
print("=" * 60)
print("""
# Visualize PCA transformation
plt.figure(figsize=(12, 5))

# Original data (first 2 dimensions)
plt.subplot(1, 2, 1)
plt.scatter(student_data[:, 0], student_data[:, 1], alpha=0.6)
plt.xlabel('Math Scores')
plt.ylabel('Physics Scores')
plt.title('Original Data (2 of 4 dimensions)')
plt.grid(True, alpha=0.3)

# PCA reduced data
plt.subplot(1, 2, 2)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.6, color='red')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Reduced Data (2 dimensions)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""")

# STUDENT EXERCISE
print("\n" + "=" * 60)
print("EXERCISE 5: Try PCA with n_components=3")
print("How much variance is retained? Is it worth the extra dimension?")
print("=" * 60)
pca3 = PCA(n_components=3)
reduced3 = pca3.fit_transform(student_data)

print("Variance explained:")
for i, v in enumerate(pca3.explained_variance_ratio_, start=1):
    print(f"Component {i}: {v*100:.2f}%")

print("Total variance retained:", f"{pca3.explained_variance_ratio_.sum()*100:.2f}%")
# ==============================================================================
# PART 6: BUILD SOMETHING REAL - MINI PROJECT (30 minutes)
# ==============================================================================

print("\n\n[PART 6] MINI PROJECT: Simple Recommendation System\n")
print("=" * 60)

class SimpleRecommender:
    """
    A basic recommendation system using linear algebra concepts
    """
    def __init__(self, user_movie_matrix, movie_names):
        self.matrix = user_movie_matrix
        self.movies = movie_names
        
    def find_similar_users(self, user_id, top_n=2):
        """Find users with similar taste"""
        user_vector = self.matrix[user_id]
        similarities = []
        
        for i in range(len(self.matrix)):
            if i != user_id:
                sim = cosine_similarity(user_vector, self.matrix[i])
                similarities.append((i, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]
    
    def recommend(self, user_id, n_recommendations=2):
        """Recommend movies based on similar users"""
        # Find similar users
        similar_users = self.find_similar_users(user_id)
        
        print(f"\nRecommendations for User {user_id}:")
        print("-" * 40)
        
        # Get current user's ratings
        user_ratings = self.matrix[user_id]
        
        # Collect recommendations
        recommendations = {}
        
        for similar_user_id, similarity in similar_users:
            print(f"Similar User {similar_user_id} (similarity: {similarity:.3f})")
            
            # Find movies this similar user liked but target user hasn't seen
            similar_ratings = self.matrix[similar_user_id]
            
            for i, (user_rating, similar_rating) in enumerate(zip(user_ratings, similar_ratings)):
                if user_rating == 0 and similar_rating >= 4:  # User hasn't seen, similar user liked
                    if self.movies[i] not in recommendations:
                        recommendations[self.movies[i]] = []
                    recommendations[self.movies[i]].append(similar_rating * similarity)
        
        # Average the recommendation scores
        final_recs = {movie: np.mean(scores) for movie, scores in recommendations.items()}
        
        # Sort and return top N
        sorted_recs = sorted(final_recs.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop Recommendations:")
        for movie, score in sorted_recs[:n_recommendations]:
            print(f"  • {movie} (score: {score:.2f})")
        
        return sorted_recs[:n_recommendations]


recommender = SimpleRecommender(user_movie_matrix, movies)


print("\nTesting the Recommender System:")
recommender.recommend(user_id=0, n_recommendations=3)

# FINAL CHALLENGE
print("\n" + "=" * 60)
print("FINAL CHALLENGE:")
print("1. Add more users and movies to the matrix")
print("2. Improve the recommendation algorithm")
print("3. Add a confidence score for recommendations")
print("4. Handle edge cases (what if no similar users?)")
print("=" * 60)

# FINAL CHALLENGE SOLUTION

extended_user_movie_matrix = np.array([
    [5, 3, 0, 4, 2, 0],   
    [4, 0, 0, 5, 1, 3],   
    [1, 4, 5, 2, 5, 4],   
    [5, 3, 1, 4, 3, 0],   
    [0, 5, 5, 1, 4, 2],   
    [3, 4, 2, 0, 5, 5],   
    [4, 1, 3, 4, 0, 4],   
])

extended_movies = ['Action', 'Comedy', 'Horror', 'Sci-Fi', 'Romance', 'Thriller']


class AdvancedRecommender(SimpleRecommender):
    """ Extended version with confidence score + edge cases """

    def recommend(self, user_id, n_recommendations=3):
        similar_users = self.find_similar_users(user_id, top_n=3)

        if not similar_users:
            print("⚠ No similar users found. Cannot recommend.")
            return []

        print(f"\nAdvanced Recommendations for User {user_id}:")
        print("-" * 50)

        user_ratings = self.matrix[user_id]
        recommendations = {}

        for sim_user, similarity in similar_users:
            if similarity <= 0:
                continue  

            other_ratings = self.matrix[sim_user]

            for idx, (u_rate, s_rate) in enumerate(zip(user_ratings, other_ratings)):
                if u_rate == 0 and s_rate >= 4:
                    weighted_score = s_rate * similarity
                    confidence = similarity ** 2     

                    if self.movies[idx] not in recommendations:
                        recommendations[self.movies[idx]] = []

                    recommendations[self.movies[idx]].append((weighted_score, confidence))

        if not recommendations:
            print("⚠ No movies to recommend based on similar users.")
            return []

       
        final_scores = {}
        for movie, score_list in recommendations.items():
            avg_score = np.mean([s for s, c in score_list])
            avg_confidence = np.mean([c for s, c in score_list])
            final_scores[movie] = (avg_score, avg_confidence)

       
        sorted_movies = sorted(
            final_scores.items(),
            key=lambda x: (x[1][0], x[1][1]),
            reverse=True
        )

       
        print("\nTop Movie Recommendations:")
        print("(Format: Score | Confidence)")
        for movie, (score, conf) in sorted_movies[:n_recommendations]:
            print(f" • {movie}: {score:.2f} | confidence = {conf:.2f}")

        return sorted_movies[:n_recommendations]


adv_rec = AdvancedRecommender(extended_user_movie_matrix, extended_movies)
adv_rec.recommend(0, n_recommendations=3)

# ==============================================================================
# SUMMARY AND NEXT STEPS
# ==============================================================================

print("\n\n" + "=" * 60)
print("SESSION COMPLETE! 🎉")
print("=" * 60)

print("\nWhat You've Learned:")
print("✓ Vectors represent data points")
print("✓ Dot product measures similarity")
print("✓ Matrices organize multiple data points")
print("✓ Matrix multiplication transforms data")
print("✓ PCA reduces dimensions while keeping information")
print("✓ Built a working recommendation system!")

print("\nWhere This Shows Up in ML:")
print("• Neural Networks: Matrix multiplication everywhere")
print("• Recommendation Systems: Similarity measures")
print("• Image Processing: Matrix transformations")
print("• Feature Engineering: PCA and dimensionality reduction")
print("• Data Preprocessing: Normalization and scaling")

print("\nNext Session: PROBABILITY")
print("We'll learn how ML makes predictions under uncertainty!")
print("=" * 60)
