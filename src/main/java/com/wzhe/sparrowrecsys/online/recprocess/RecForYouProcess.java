package com.wzhe.sparrowrecsys.online.recprocess;

import com.wzhe.sparrowrecsys.online.datamanager.DataManager;
import com.wzhe.sparrowrecsys.online.datamanager.Movie;
import com.wzhe.sparrowrecsys.online.datamanager.RedisClient;
import com.wzhe.sparrowrecsys.online.datamanager.User;
import com.wzhe.sparrowrecsys.online.util.Config;
import com.wzhe.sparrowrecsys.online.util.Utility;

import java.util.*;

/**
 * Recommendation process of similar movies
 */

public class RecForYouProcess {

    /**
     * get recommendation movie list
     * @param userId input user id
     * @param size  size of similar items
     * @param model model used for calculating similarity
     * @return  list of similar movies
     */
    public static List<Movie> getRecList(int userId, int size, String model){
        User user = DataManager.getInstance().getUserById(userId);
        if (null == user){
            return new ArrayList<>();
        }
        final int CANDIDATE_SIZE = 800;
        List<Movie> candidates = DataManager.getInstance().getMovies(CANDIDATE_SIZE, "rating");

        //load user emb from redis if data source is redis
        if (Config.EMB_DATA_SOURCE.equals(Config.DATA_SOURCE_REDIS)){
            String userEmbKey = "uEmb:" + userId;
            String userEmb = RedisClient.getInstance().get(userEmbKey);
            if (null != userEmb){
                user.setEmb(Utility.parseEmbStr(userEmb));
            }
        }

        List<Movie> rankedList = ranker(user, candidates, model);

        if (rankedList.size() > size){
            return rankedList.subList(0, size);
        }
        return rankedList;
    }

    /**
     * rank candidates
     * @param user    input user
     * @param candidates    movie candidates
     * @param model     model name used for ranking
     * @return  ranked movie list
     */
    public static List<Movie> ranker(User user, List<Movie> candidates, String model){
        HashMap<Movie, Double> candidateScoreMap = new HashMap<>();
        for (Movie candidate : candidates){
            double similarity;
            switch (model){
                case "emb":
                    similarity = calculateEmbSimilarScore(user, candidate);
                    break;
                default:
                    similarity = calculateEmbSimilarScore(user, candidate);
            }
            candidateScoreMap.put(candidate, similarity);
        }
        List<Movie> rankedList = new ArrayList<>();
        candidateScoreMap.entrySet().stream().sorted(Map.Entry.comparingByValue(Comparator.reverseOrder())).forEach(m -> rankedList.add(m.getKey()));
        return rankedList;
    }

    /**
     * function to calculate similarity score based on embedding
     * @param user     input user
     * @param candidate candidate movie
     * @return  similarity score
     */
    public static double calculateEmbSimilarScore(User user, Movie candidate){
        if (null == user || null == candidate || null == user.getEmb()){
            return -1;
        }
        return user.getEmb().calculateSimilarity(candidate.getEmb());
    }
}
