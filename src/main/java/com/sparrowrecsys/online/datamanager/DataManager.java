package com.sparrowrecsys.online.datamanager;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

/**
 * DataManager is an utility class, takes charge of all data loading logic.
 */

public class DataManager {
    //singleton instance
    private static volatile DataManager instance;
    HashMap<Integer, News> newsMap;
    HashMap<Integer, User> userMap;
    //genre reverse index for quick querying all movies in a genre
    HashMap<NewsNer, List<News>> nerReverseIndexMap;

    private DataManager(){
        this.newsMap = new HashMap<>();
        this.userMap = new HashMap<>();
        this.nerReverseIndexMap = new HashMap<>();
//        this.genreReverseIndexMap = new HashMap<>();
        instance = this;
    }

    public static DataManager getInstance(){
        if (null == instance){
            synchronized (DataManager.class){
                if (null == instance){
                    instance = new DataManager();
                }
            }
        }
        return instance;
    }


    public void loadData(String newsDataPath) throws Exception{
        loadNewsData(newsDataPath);
//        loadLinkData(linkDataPath);
//        loadRatingData(ratingDataPath);
//        loadMovieEmb(movieEmbPath, movieRedisKey);
//        if (Config.IS_LOAD_ITEM_FEATURE_FROM_REDIS){
//            loadMovieFeatures("mf:");
//        }
//
//        loadUserEmb(userEmbPath, userRedisKey);
    }

    private void loadNewsData(String newsDataPath) throws Exception {
        System.out.println("Loading news data from" + newsDataPath + " ...");
        boolean skipFirstLine = true;
        //JSON parser object to parse read file
        JSONParser jsonParser = new JSONParser();
        ArrayList<JSONObject> json=new ArrayList<JSONObject>();


        try (FileReader reader = new FileReader(newsDataPath)) {
            BufferedReader bufferedReader = new BufferedReader(reader);
            String line = null;
            while((line = bufferedReader.readLine()) != null) {
                JSONObject obj = (JSONObject) new JSONParser().parse(line);
                json.add(obj);
//                System.out.println((String)obj.get("Sensor_ID")+":"+
//                        (String)obj.get("Date"));
            }
            // Always close files.
            bufferedReader.close();

            for (JSONObject obj : json) {
                try {
                    News news = parseNewsObject(obj);
                    newsMap.put(news.getNewsId(), news);
                } catch (java.text.ParseException e) {
                    e.printStackTrace();
                }
            }
        } catch (ParseException | IOException e) {
            e.printStackTrace();
        }
    }

    private News parseNewsObject(JSONObject employee) throws java.text.ParseException {
        News news = new News();
        JSONObject newsObject = (JSONObject) employee.get("Item");

        // Get url
        String url = (String) ((JSONObject) newsObject.get("article_url")).get("S");
        news.setNewsUrl(url);

        // Set newsId
        if (newsMap.size() >= Integer.MAX_VALUE) throw new RuntimeException();
        news.setNewsId(newsMap.size() + 1);

        String publishDate = (String) ((JSONObject) newsObject.get("date_publish")).get("S");
        news.setReleaseDate(new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").parse(publishDate));

        news.setNumUpVotes(Integer.parseInt((String) ((JSONObject) newsObject.get("upvote")).get("N")));

        news.setSourceDomain((String) ((JSONObject) newsObject.get("source_domain")).get("S"));

        // get topics
        JSONArray topics = (JSONArray) ((JSONObject) newsObject.get("topics")).get("L");
        Iterator i = topics.iterator();
        List<String> topicList = new ArrayList<>();
        while (i.hasNext()) {
            topicList.add((String) ((JSONObject) i.next()).get("S"));
        }
        news.setTopics(topicList);

        news.setNumDownVotes(Integer.parseInt((String) ((JSONObject) newsObject.get("downvote")).get("N")));

        String sentiment = (String) ((JSONObject) newsObject.get("sentiment")).get("S");
        double polarity = Double.parseDouble(sentiment.substring(sentiment.indexOf("p") + 10, sentiment.indexOf(",")));
        double subjectivity = Double.parseDouble(sentiment.substring(sentiment.indexOf("s") + 14, sentiment.indexOf("}")));
        news.setPolarity(polarity);
        news.setSubjectivity(subjectivity);

        String category = (String) ((JSONObject) newsObject.get("category")).get("S");
        news.setCategory(NewsCate.valueOf(category.toUpperCase()));

        // Get NERs
        List<NewsNer> ners = new ArrayList<>();
        String nerStr = (String) ((JSONObject) newsObject.get("named_entities")).get("S");
        nerStr = nerStr.substring(10);
        while (nerStr.contains("\"text\"")) {
            String text = nerStr.substring(nerStr.indexOf("\"text\"") + 8, nerStr.indexOf("\",\"label\""));
            String label = nerStr.substring(nerStr.indexOf("\"label\"") + 9, nerStr.indexOf(("\",\"lemma\":")));
            int count = Integer.parseInt(nerStr.substring(nerStr.indexOf("\"count\":") + 8, nerStr.indexOf("}")));
            ners.add(new NewsNer(text, label, count));
            nerStr = nerStr.substring(nerStr.indexOf("},{") + 3);
        }
        for (NewsNer ner : ners) {
            addNews2NerIndex(news, ner);
        }
        news.setNers(ners);

        // Get title
        news.setNewsUrl((String) ((JSONObject) newsObject.get("article_url")).get("S"));

        // Get author
        JSONArray authors = (JSONArray) ((JSONObject) newsObject.get("topics")).get("L");
        Iterator authorsItr = authors.iterator();
        List<String> authorList = new ArrayList<>();
        while (authorsItr.hasNext()) {
            authorList.add((String) ((JSONObject) authorsItr.next()).get("S"));
        }
        news.setAuthors(authorList);
        return news;
    }

    private void addNews2NerIndex(News news, NewsNer ner) {
        if (!this.nerReverseIndexMap.containsKey(ner)){
            this.nerReverseIndexMap.put(ner, new ArrayList<>());
        }
        this.nerReverseIndexMap.get(ner).add(news);
    }

    //get movies by genre, and order the movies by sortBy method
    public List<News> getNewsByNer(NewsNer ner, int size, String sortBy){
        if (null != ner){
            List<News> newsList = new ArrayList<>(this.nerReverseIndexMap.get(ner));
            switch (sortBy){
                case "popularity":newsList.sort((n1, n2) -> Double.compare(n2.getPopularity(), n1.getPopularity()));break;
                case "releaseDate": newsList.sort((n1, n2) -> n2.getReleaseDate().compareTo(n1.getReleaseDate()));break;
                default:
            }

            if (newsList.size() > size){
                return newsList.subList(0, size);
            }
            return newsList;
        }
        return null;
    }

    //get top N movies order by sortBy method
    public List<News> getNews(int size, String sortBy){
        List<News> newsList = new ArrayList<>(newsMap.values());
        switch (sortBy){
            case "popularity":newsList.sort((n1, n2) -> Double.compare(n2.getPopularity(), n1.getPopularity()));break;
            case "releaseDate": newsList.sort((n1, n2) -> n2.getReleaseDate().compareTo(n1.getReleaseDate()));break;
            default:
        }
        if (newsList.size() > size){
            return newsList.subList(0, size);
        }
        return newsList;
    }

    //get movie object by movie id
    public News getNewsById(int newsId){
        return this.newsMap.get(newsId);
    }
//
//    //get user object by user id
//    public User getUserById(int userId){
//        return this.userMap.get(userId);
//    }
}
