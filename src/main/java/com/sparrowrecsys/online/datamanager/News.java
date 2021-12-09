package com.sparrowrecsys.online.datamanager;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import com.sparrowrecsys.online.model.Embedding;

import java.sql.Time;
import java.util.Date;
import java.util.List;
import java.util.Map;

public class News {
    int newsId;
    String newsUrl;
    Date releaseDate;
    String title;
    List<String> topics;
    String sourceDomain;
    int numUpVotes;
    int numDownVotes;
    int numComments;
    int numForwards;
    double polarity;
    double subjectivity;
    NewsCate category;
    List<NewsNer> ners;
    List<String> authors;

    public List<String> getAuthors() {
        return authors;
    }

    public void setAuthors(List<String> authors) {
        this.authors = authors;
    }

    //embedding of the news
    @JsonIgnore
    Embedding emb;

    @JsonIgnore
    Map<String, String> newsFeatures;

    public double getPopularity() {
        return this.getNumUpVotes() - this.getNumDownVotes() + 2 * this.getNumComments();
    }

    public void setNewsId(int newsId) {
        this.newsId = newsId;
    }

    public void setNewsUrl(String newsUrl) {
        this.newsUrl = newsUrl;
    }

    public void setReleaseDate(Date releaseTime) {
        this.releaseDate = releaseTime;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public void setTopics(List<String> topics) {
        this.topics = topics;
    }

    public void setSourceDomain(String sourceDomain) {
        this.sourceDomain = sourceDomain;
    }

    public void setNumUpVotes(int numUpVotes) {
        this.numUpVotes = numUpVotes;
    }

    public void setNumDownVotes(int numDownVotes) {
        this.numDownVotes = numDownVotes;
    }

    public void setNumComments(int numComments) {
        this.numComments = numComments;
    }

    public void setNumForwards(int numForwards) {
        this.numForwards = numForwards;
    }

    public void setPolarity(double polarity) {
        this.polarity = polarity;
    }

    public void setSubjectivity(double subjectivity) {
        this.subjectivity = subjectivity;
    }

    public void setCategory(NewsCate category) {
        this.category = category;
    }

    public void setNers(List<NewsNer> ners) {
        this.ners = ners;
    }

    public void setEmb(Embedding emb) {
        this.emb = emb;
    }

    public void setNewsFeatures(Map<String, String> newsFeatures) {
        this.newsFeatures = newsFeatures;
    }

    public int getNewsId() {
        return newsId;
    }

    public String getNewsUrl() {
        return newsUrl;
    }

    public Date getReleaseDate() {
        return releaseDate;
    }

    public String getTitle() {
        return title;
    }

    public List<String> getTopics() {
        return topics;
    }

    public String getSourceDomain() {
        return sourceDomain;
    }

    public int getNumUpVotes() {
        return numUpVotes;
    }

    public int getNumDownVotes() {
        return numDownVotes;
    }

    public int getNumComments() {
        return numComments;
    }

    public int getNumForwards() {
        return numForwards;
    }

    public double getPolarity() {
        return polarity;
    }

    public double getSubjectivity() {
        return subjectivity;
    }

    public NewsCate getCategory() {
        return category;
    }

    public List<NewsNer> getNers() {
        return ners;
    }

    public Embedding getEmb() {
        return emb;
    }

    public Map<String, String> getNewsFeatures() {
        return newsFeatures;
    }
}
