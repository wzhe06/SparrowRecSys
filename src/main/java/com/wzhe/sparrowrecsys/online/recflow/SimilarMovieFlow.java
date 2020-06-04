package com.wzhe.sparrowrecsys.online.recflow;

import com.wzhe.sparrowrecsys.online.datamanager.DataManager;
import com.wzhe.sparrowrecsys.online.datamanager.Movie;

import javax.swing.plaf.multi.MultiViewportUI;
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
        if (candidateMap.containsKey(movie.getMovieId())){
            candidateMap.remove(movie.getMovieId());
        }
        return new ArrayList<>(candidateMap.values());
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
