selected_lines_anonymized.json.bz -- Gzipped JSON file containing usage logs from deepbeat.org
                                     "Suggest Rhyming Line" functionality.

If you want to have more details about data collection or want to use this dataset, please
check/cite:

Malmi, E., Takala, P., Toivonen, H., Raiko, T., Gionis, A. DopeLearning: A Computational Approach
to Rap Lyrics Generation. In Proceeding of the 22nd ACM SIGKDD Conference on Knowledge Discovery
and Data Mining, 2016.

The JSON file contains a list of user selections of the form:

{
  'lineCandidates': [
      {'line': <candidate next line text>,
       'meta': {'artist': <artist name>, 'track': <track name>},
       'nn_score': <relevance score using FastFeatsNN5>,
       'score': <relevance score using FastFeats>},
        ... 19 other candidate next lines ...
  ],
  'queryLines': [<1-5 previous lines>],
  'selectedLine': <index of the candidate user has chosen from 'lineCandidates'>,
  'time': <seconds from the first logged selection user has made>,
  'userID': <anonymized user ID based on the combination of IP address and user agent>
}

Selections including manually defined queryLines have been removed from the dataset to preverse
user privacy.
