package com.sparrowrecsys.online.recprocess;

import com.sparrowrecsys.online.datamanager.DataManager;
import com.sparrowrecsys.online.datamanager.News;
import com.sparrowrecsys.online.datamanager.NewsNer;
import scala.reflect.internal.Trees;

import java.util.*;

public class SimilarNewsProcess {

    /**
     * multiple-retrieval candidate generation method
     * @param news input news object
     * @return news candidates
     */

    public static List<News> getRecList(int newsId, int size, String model){
        News news = DataManager.getInstance().getNewsById(newsId);
        if (null == news){
            return new ArrayList<>();
        }
        List<News> candidates = multipleRetrievalCandidates(news);
        List<News> rankedList = ranker(news, candidates, model);

        if (rankedList.size() > size){
            return rankedList.subList(0, size);
        }
        return rankedList;
    }

    public static List<News> multipleRetrievalCandidates(News news){
        if (null == news){
            return null;
        }

        HashSet<NewsNer> ners = new HashSet<>(news.getNers());

        HashMap<Integer, News> candidateMap = new HashMap<>();
        for (NewsNer ner : ners){
            List<News> oneCandidates = DataManager.getInstance().getNewsByNer(ner, 20, "releaseDate");
            for (News candidate : oneCandidates){
                candidateMap.put(candidate.getNewsId(), candidate);
            }
        }

        List<News> highRatingCandidates = DataManager.getInstance().getNews(100, "popularity");
        for (News candidate : highRatingCandidates){
            candidateMap.put(candidate.getNewsId(), candidate);
        }

        List<News> latestCandidates = DataManager.getInstance().getNews(100, "releaseDate");
        for (News candidate : latestCandidates){
            candidateMap.put(candidate.getNewsId(), candidate);
        }

        candidateMap.remove(news.getNewsId());
        return new ArrayList<>(candidateMap.values());
    }


    /**
     * rank candidates
     * @param news    input movie
     * @param candidates    movie candidates
     * @param model     model name used for ranking
     * @return  ranked movie list
     */
    public static List<News> ranker(News news, List<News> candidates, String model){
        HashMap<News, Double> candidateScoreMap = new HashMap<>();
        for (News candidate : candidates){
            double similarity;
            switch (model){
//                case "emb":
//                    similarity = calculateEmbSimilarScore(news, candidate);
//                    break;
                default:
                    similarity = calculateSimilarScore(news, candidate);
            }
            candidateScoreMap.put(candidate, similarity);
        }
        List<News> rankedList = new ArrayList<>();
        candidateScoreMap.entrySet().stream().sorted(Map.Entry.comparingByValue(Comparator.reverseOrder())).forEach(m -> rankedList.add(m.getKey()));
        return rankedList;
    }


    /**
     * function to calculate similarity score
     * @param news     input movie
     * @param candidate candidate movie
     * @return  similarity score
     */
    public static double calculateSimilarScore(News news, News candidate){
        int sameNerCount = 0;
        for (NewsNer ner : news.getNers()){
            if (candidate.getNers().contains(ner)){
                sameNerCount++;
            }
        }
        double genreSimilarity = (double)sameNerCount / (news.getNers().size() + candidate.getNers().size()) / 2;
        double ratingScore = candidate.getPolarity() / 5; // TODO: need to scale

        double similarityWeight = 0.7; // TODO: need tuning
        double ratingScoreWeight = 0.3;

        return genreSimilarity * similarityWeight + ratingScore * ratingScoreWeight;
    }

}
