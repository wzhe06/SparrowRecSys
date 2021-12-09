package com.sparrowrecsys.online.datamanager;

<<<<<<< HEAD
import com.sparrowrecsys.online.util.Config;
import com.sparrowrecsys.online.util.Utility;

import java.io.File;
import java.util.*;

=======
import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;

import org.apache.hadoop.util.hash.Hash;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;


>>>>>>> Ethan
/**
 * DataManager is an utility class, takes charge of all data loading logic.
 */

public class DataManager {
    //singleton instance
    private static volatile DataManager instance;
<<<<<<< HEAD
    HashMap<Integer, Movie> movieMap;
    HashMap<Integer, User> userMap;
    //genre reverse index for quick querying all movies in a genre
    HashMap<String, List<Movie>> genreReverseIndexMap;

    private DataManager(){
        this.movieMap = new HashMap<>();
        this.userMap = new HashMap<>();
        this.genreReverseIndexMap = new HashMap<>();
=======
    HashMap<Integer, News> newsMap;
    HashMap<Integer, User> userMap;
    //genre reverse index for quick querying all movies in a genre
    HashMap<String, List<News>> nerReverseIndexMap;

    private DataManager(){
        this.newsMap = new HashMap<>();
        this.userMap = new HashMap<>();
        this.nerReverseIndexMap = new HashMap<>();
//        this.genreReverseIndexMap = new HashMap<>();
>>>>>>> Ethan
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

<<<<<<< HEAD
    //load data from file system including movie, rating, link data and model data like embedding vectors.
    public void loadData(String movieDataPath, String linkDataPath, String ratingDataPath, String movieEmbPath, String userEmbPath, String movieRedisKey, String userRedisKey) throws Exception{
        loadMovieData(movieDataPath);
        loadLinkData(linkDataPath);
        loadRatingData(ratingDataPath);
        loadMovieEmb(movieEmbPath, movieRedisKey);
        if (Config.IS_LOAD_ITEM_FEATURE_FROM_REDIS){
            loadMovieFeatures("mf:");
        }

        loadUserEmb(userEmbPath, userRedisKey);
    }

    //load movie data from movies.csv
    private void loadMovieData(String movieDataPath) throws Exception{
        System.out.println("Loading movie data from " + movieDataPath + " ...");
        boolean skipFirstLine = true;
        try (Scanner scanner = new Scanner(new File(movieDataPath))) {
            while (scanner.hasNextLine()) {
                String movieRawData = scanner.nextLine();
                if (skipFirstLine){
                    skipFirstLine = false;
                    continue;
                }
                String[] movieData = movieRawData.split(",");
                if (movieData.length == 3){
                    Movie movie = new Movie();
                    movie.setMovieId(Integer.parseInt(movieData[0]));
                    int releaseYear = parseReleaseYear(movieData[1].trim());
                    if (releaseYear == -1){
                        movie.setTitle(movieData[1].trim());
                    }else{
                        movie.setReleaseYear(releaseYear);
                        movie.setTitle(movieData[1].trim().substring(0, movieData[1].trim().length()-6).trim());
                    }
                    String genres = movieData[2];
                    if (!genres.trim().isEmpty()){
                        String[] genreArray = genres.split("\\|");
                        for (String genre : genreArray){
                            movie.addGenre(genre);
                            addMovie2GenreIndex(genre, movie);
                        }
                    }
                    this.movieMap.put(movie.getMovieId(), movie);
                }
            }
        }
        System.out.println("Loading movie data completed. " + this.movieMap.size() + " movies in total.");
    }

    //load movie embedding
    private void loadMovieEmb(String movieEmbPath, String embKey) throws Exception{
        if (Config.EMB_DATA_SOURCE.equals(Config.DATA_SOURCE_FILE)) {
            System.out.println("Loading movie embedding from " + movieEmbPath + " ...");
            int validEmbCount = 0;
            try (Scanner scanner = new Scanner(new File(movieEmbPath))) {
                while (scanner.hasNextLine()) {
                    String movieRawEmbData = scanner.nextLine();
                    String[] movieEmbData = movieRawEmbData.split(":");
                    if (movieEmbData.length == 2) {
                        Movie m = getMovieById(Integer.parseInt(movieEmbData[0]));
                        if (null == m) {
                            continue;
                        }
                        m.setEmb(Utility.parseEmbStr(movieEmbData[1]));
                        validEmbCount++;
                    }
                }
            }
            System.out.println("Loading movie embedding completed. " + validEmbCount + " movie embeddings in total.");
        }else{
            System.out.println("Loading movie embedding from Redis ...");
            Set<String> movieEmbKeys = RedisClient.getInstance().keys(embKey + "*");
            int validEmbCount = 0;
            for (String movieEmbKey : movieEmbKeys){
                String movieId = movieEmbKey.split(":")[1];
                Movie m = getMovieById(Integer.parseInt(movieId));
                if (null == m) {
                    continue;
                }
                m.setEmb(Utility.parseEmbStr(RedisClient.getInstance().get(movieEmbKey)));
                validEmbCount++;
            }
            System.out.println("Loading movie embedding completed. " + validEmbCount + " movie embeddings in total.");
        }
    }

    //load movie features
    private void loadMovieFeatures(String movieFeaturesPrefix) throws Exception{
        System.out.println("Loading movie features from Redis ...");
        Set<String> movieFeaturesKeys = RedisClient.getInstance().keys(movieFeaturesPrefix + "*");
        int validFeaturesCount = 0;
        for (String movieFeaturesKey : movieFeaturesKeys){
            String movieId = movieFeaturesKey.split(":")[1];
            Movie m = getMovieById(Integer.parseInt(movieId));
            if (null == m) {
                continue;
            }
            m.setMovieFeatures(RedisClient.getInstance().hgetAll(movieFeaturesKey));
            validFeaturesCount++;
        }
        System.out.println("Loading movie features completed. " + validFeaturesCount + " movie features in total.");
    }

    //load user embedding
    private void loadUserEmb(String userEmbPath, String embKey) throws Exception{
        if (Config.EMB_DATA_SOURCE.equals(Config.DATA_SOURCE_FILE)) {
            System.out.println("Loading user embedding from " + userEmbPath + " ...");
            int validEmbCount = 0;
            try (Scanner scanner = new Scanner(new File(userEmbPath))) {
                while (scanner.hasNextLine()) {
                    String userRawEmbData = scanner.nextLine();
                    String[] userEmbData = userRawEmbData.split(":");
                    if (userEmbData.length == 2) {
                        User u = getUserById(Integer.parseInt(userEmbData[0]));
                        if (null == u) {
                            continue;
                        }
                        u.setEmb(Utility.parseEmbStr(userEmbData[1]));
                        validEmbCount++;
                    }
                }
            }
            System.out.println("Loading user embedding completed. " + validEmbCount + " user embeddings in total.");
        }
    }

    //parse release year
    private int parseReleaseYear(String rawTitle){
        if (null == rawTitle || rawTitle.trim().length() < 6){
            return -1;
        }else{
            String yearString = rawTitle.trim().substring(rawTitle.length()-5, rawTitle.length()-1);
            try{
                return Integer.parseInt(yearString);
            }catch (NumberFormatException exception){
                return -1;
            }
        }
    }

    //load links data from links.csv
    private void loadLinkData(String linkDataPath) throws Exception{
        System.out.println("Loading link data from " + linkDataPath + " ...");
        int count = 0;
        boolean skipFirstLine = true;
        try (Scanner scanner = new Scanner(new File(linkDataPath))) {
            while (scanner.hasNextLine()) {
                String linkRawData = scanner.nextLine();
                if (skipFirstLine){
                    skipFirstLine = false;
                    continue;
                }
                String[] linkData = linkRawData.split(",");
                if (linkData.length == 3){
                    int movieId = Integer.parseInt(linkData[0]);
                    Movie movie = this.movieMap.get(movieId);
                    if (null != movie){
                        count++;
                        movie.setImdbId(linkData[1].trim());
                        movie.setTmdbId(linkData[2].trim());
                    }
                }
            }
        }
        System.out.println("Loading link data completed. " + count + " links in total.");
    }

    //load ratings data from ratings.csv
    private void loadRatingData(String ratingDataPath) throws Exception{
        System.out.println("Loading rating data from " + ratingDataPath + " ...");
        boolean skipFirstLine = true;
        int count = 0;
        try (Scanner scanner = new Scanner(new File(ratingDataPath))) {
            while (scanner.hasNextLine()) {
                String ratingRawData = scanner.nextLine();
                if (skipFirstLine){
                    skipFirstLine = false;
                    continue;
                }
                String[] linkData = ratingRawData.split(",");
                if (linkData.length == 4){
                    count ++;
                    Rating rating = new Rating();
                    rating.setUserId(Integer.parseInt(linkData[0]));
                    rating.setMovieId(Integer.parseInt(linkData[1]));
                    rating.setScore(Float.parseFloat(linkData[2]));
                    rating.setTimestamp(Long.parseLong(linkData[3]));
                    Movie movie = this.movieMap.get(rating.getMovieId());
                    if (null != movie){
                        movie.addRating(rating);
                    }
                    if (!this.userMap.containsKey(rating.getUserId())){
                        User user = new User();
                        user.setUserId(rating.getUserId());
                        this.userMap.put(user.getUserId(), user);
                    }
                    this.userMap.get(rating.getUserId()).addRating(rating);
                }
            }
        }

        System.out.println("Loading rating data completed. " + count + " ratings in total.");
    }

    //add movie to genre reversed index
    private void addMovie2GenreIndex(String genre, Movie movie){
        if (!this.genreReverseIndexMap.containsKey(genre)){
            this.genreReverseIndexMap.put(genre, new ArrayList<>());
        }
        this.genreReverseIndexMap.get(genre).add(movie);
    }

    //get movies by genre, and order the movies by sortBy method
    public List<Movie> getMoviesByGenre(String genre, int size, String sortBy){
        if (null != genre){
            List<Movie> movies = new ArrayList<>(this.genreReverseIndexMap.get(genre));
            switch (sortBy){
                case "rating":movies.sort((m1, m2) -> Double.compare(m2.getAverageRating(), m1.getAverageRating()));break;
                case "releaseYear": movies.sort((m1, m2) -> Integer.compare(m2.getReleaseYear(), m1.getReleaseYear()));break;
                default:
            }

            if (movies.size() > size){
                return movies.subList(0, size);
            }
            return movies;
=======

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
                    putNews2Redis(news);
                } catch (java.text.ParseException e) {
                    e.printStackTrace();
                }
            }
//            List<Integer> testList = getNewsIdByNer("Nets", -1);
//            for (int i : testList) {
//                System.out.printf("%d ", i);
//            }
//            System.out.printf("\n");

//            System.out.println(nerReverseIndexMap.keySet().size());
            int maxSize = 0;
            for (String nerText : nerReverseIndexMap.keySet()) {
                maxSize = Math.max(maxSize, nerReverseIndexMap.get(nerText).size());
            }
//            System.out.println(maxSize);
        } catch (ParseException | IOException e) {
            e.printStackTrace();
        }
    }

//    private void indexNewsMap2Redis(HashMap<Integer, News> newsMap) {
//        for (int newsId : newsMap.keySet()) {
//            News news = newsMap.get(newsId);
//            Long time = news.releaseDate.getTime();
//            for (NewsNer)
//        }
//    }

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
        if (!this.nerReverseIndexMap.containsKey(ner.getText())){
            this.nerReverseIndexMap.put(ner.getText(), new ArrayList<>());
        }
        this.nerReverseIndexMap.get(ner.getText()).add(news);
    }

    private void putNews2Redis(News news) {
        Long time = news.getReleaseDate().getTime();
        for (NewsNer ner : news.ners) {
            String key = "NER:" + ner.getText();
            String value = String.valueOf(ner.getCount()) + ":" + String.valueOf(news.getNewsId());
            RedisClient.getInstance().zadd(key, (double) time, value);
        }
    }

    public List<Integer> getNewsIdByNer(String nerText, int expireDay) {
        String key = "NER:" + nerText;
        long DAY_IN_MS = 1000 * 60 * 60 * 24;
        Set<String> set;
        if (expireDay > 0) {
            set = RedisClient.getInstance().zrevrange(key, System.currentTimeMillis() - (expireDay * DAY_IN_MS), -1);
        } else {
            set = RedisClient.getInstance().zrevrange(key, 0, -1);
        }

        List<Integer> newsIdList = new ArrayList<>();
        for (String countNewsId : set) {
            int newsId = Integer.parseInt(countNewsId.split(":")[1]);
            newsIdList.add(newsId);
        }
        return newsIdList;
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
>>>>>>> Ethan
        }
        return null;
    }

    //get top N movies order by sortBy method
<<<<<<< HEAD
    public List<Movie> getMovies(int size, String sortBy){
            List<Movie> movies = new ArrayList<>(movieMap.values());
            switch (sortBy){
                case "rating":movies.sort((m1, m2) -> Double.compare(m2.getAverageRating(), m1.getAverageRating()));break;
                case "releaseYear": movies.sort((m1, m2) -> Integer.compare(m2.getReleaseYear(), m1.getReleaseYear()));break;
                default:
            }

            if (movies.size() > size){
                return movies.subList(0, size);
            }
            return movies;
    }

    //get movie object by movie id
    public Movie getMovieById(int movieId){
        return this.movieMap.get(movieId);
    }

    //get user object by user id
    public User getUserById(int userId){
        return this.userMap.get(userId);
    }
=======
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
>>>>>>> Ethan
}
