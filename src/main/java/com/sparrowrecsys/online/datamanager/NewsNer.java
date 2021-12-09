package com.sparrowrecsys.online.datamanager;



public class NewsNer {
    private String text;
    private int count;
    private String label;

    public NewsNer(String text, String label, int count) {
        this.text = text;
        this.label = label;
        this.count = count;
    }

    public String getText() {
        return text;
    }
    public int getCount() {
        return count;
    }
    public String getLabel() {
        return label;
    }

    public void setCount(int count) {
        this.count = count;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public void setText(String text) {
        this.text = text;
    }
}
