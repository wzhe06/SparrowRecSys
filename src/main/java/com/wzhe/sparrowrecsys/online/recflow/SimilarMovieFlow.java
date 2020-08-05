package com.wzhe.sparrowrecsys.online.recflow;

import com.wzhe.sparrowrecsys.online.datamanager.DataManager;
import com.wzhe.sparrowrecsys.online.datamanager.Movie;
import com.wzhe.sparrowrecsys.online.datamanager.User;

import java.util.*;

public class SimilarMovieFlow {

    public static List<Movie> getRecList(int movieId, int size, String model){
        Movie movie = DataManager.getInstance().getMovieById(movieId);
        if (null == movie){
            return new ArrayList<>();
        }
        List<Movie> candidates = candidateGenerator(movie);
        List<Movie> rankedList = ranker(movie, candidates);

        if (rankedList.size() > size){
            return rankedList.subList(0, size);
        }
        return rankedList;
    }

    public static List<Movie> candidateGenerator(Movie movie){
        ArrayList<Movie> candidates = new ArrayList<>();
        HashMap<Integer, Movie> candidateMap = new HashMap<>();
        for (String genre : movie.getGenres()){
            List<Movie> oneCandidates = DataManager.getInstance().getMoviesByGenre(genre, 100, "rating");
            for (Movie candidate : oneCandidates){
                candidateMap.put(candidate.getMovieId(), candidate);
            }
        }
        candidateMap.remove(movie.getMovieId());
        return new ArrayList<>(candidateMap.values());
    }


    public static List<Movie> multipleRetrievalCandidates(List<Movie> userHistory){
        HashSet<String> genres = new HashSet<>();
        for (Movie movie : userHistory){
            genres.addAll(movie.getGenres());
        }

        HashMap<Integer, Movie> candidateMap = new HashMap<>();
        for (String genre : genres){
            List<Movie> oneCandidates = DataManager.getInstance().getMoviesByGenre(genre, 20, "rating");
            for (Movie candidate : oneCandidates){
                candidateMap.put(candidate.getMovieId(), candidate);
            }
        }

        List<Movie> highRatingCandidates = DataManager.getInstance().getMovies(100, "rating");
        for (Movie candidate : highRatingCandidates){
            candidateMap.put(candidate.getMovieId(), candidate);
        }

        List<Movie> latestCandidates = DataManager.getInstance().getMovies(100, "releaseYear");
        for (Movie candidate : latestCandidates){
            candidateMap.put(candidate.getMovieId(), candidate);
        }

        for (Movie movie : userHistory){
            candidateMap.remove(movie.getMovieId());
        }

        return new ArrayList<>(candidateMap.values());
    }


    public static List<Movie> retrievalCandidatesByEmbedding(User user, int size){
        if (null == user){
            return null;
        }
        double[] userEmbedding = DataManager.getInstance().getUserEmbedding(user.getUserId(), "item2vec");
        if (null == userEmbedding){
            return null;
        }
        List<Movie> allCandidates = DataManager.getInstance().getMovies(10000, "rating");
        HashMap<Movie,Double> movieScoreMap = new HashMap<>();
        for (Movie candidate : allCandidates){
            double[] itemEmbedding = DataManager.getInstance().getItemEmbedding(candidate.getMovieId(), "item2vec");
            double similarity = calculateEmbeddingSimilarity(userEmbedding, itemEmbedding);
            movieScoreMap.put(candidate, similarity);
        }

        List<Map.Entry<Movie,Double>> movieScoreList = new ArrayList<>(movieScoreMap.entrySet());
        movieScoreList.sort(Map.Entry.comparingByValue());

        List<Movie> candidates = new ArrayList<>();
        for (Map.Entry<Movie,Double> movieScoreEntry : movieScoreList){
            candidates.add(movieScoreEntry.getKey());
        }

        return candidates.subList(0, Math.min(candidates.size(), size));
    }

    private static double calculateEmbeddingSimilarity(double[] embedding1, double[] embedding2){
        if (null == embedding1 || null == embedding2 || embedding1.length != embedding2.length){
            return 0d;
        }
        double dotProduct = 0;
        for (int i = 0; i < embedding1.length; i++){
            dotProduct += embedding1[i] * embedding2[i];
        }
        return dotProduct;
    }




    public static List<Movie> ranker(Movie movie, List<Movie> candidates){
        HashMap<Movie, Double> candidateScoreMap = new HashMap<>();
        for (Movie candidate : candidates){
            candidateScoreMap.put(candidate, calculateSimilarScore(movie, candidate));
        }
        List<Movie> rankedList = new ArrayList<>();
        candidateScoreMap.entrySet().stream().sorted(Map.Entry.comparingByValue(Comparator.reverseOrder())).forEach(m -> rankedList.add(m.getKey()));
        return rankedList;
    }

    public static double calculateSimilarScore(Movie movie, Movie candidate){
        int sameGenreCount = 0;
        for (String genre : movie.getGenres()){
            if (candidate.getGenres().contains(genre)){
                sameGenreCount++;
            }
        }
        double genreSimilarity = (double)sameGenreCount / (movie.getGenres().size() + candidate.getGenres().size()) / 2;
        double ratingScore = candidate.getAverageRating() / 5;

        double similarityWeight = 0.7;
        double ratingScoreWeight = 0.3;

        return genreSimilarity * similarityWeight + ratingScore * ratingScoreWeight;
    }
}
