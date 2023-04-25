using System;
using Accord.Math.Distances;
using Accord.MachineLearning;
using TorchSharp;
using System.IO;
using Accord.IO;


namespace TensorProcessor
{
    class Program
    {
        public static int[] ClusterEmbeddings(double[][] embeddings, int numClusters)
        {
            var kmeans = new KMeans(numClusters, new Euclidean())
            {
                Tolerance = 1e-5,
                MaxIterations = 100,
            };

            var clusterLabels = kmeans.Learn(embeddings);
            int[] labels = clusterLabels.Decide(embeddings);
            return labels;
        }

        public static double[][] LoadEmbeddings(string filePath)
        {
            var tensor = torch.load(filePath);
            var shape = tensor.shape;
            long rows = shape[0];
            long cols = shape[1];

            double[][] embeddings = new double[rows][];

            for (int i = 0; i < rows; i++)
            {
                embeddings[i] = new double[cols];
                for (int j = 0; j < cols; j++)
                {
                    embeddings[i][j] = tensor[i, j].ToDouble();
                }
            }

            return embeddings;
        }


        static void Main(string[] args)
        {
            //if (args.Length == 0)
            //{
            //    Console.WriteLine("Пожалуйста, укажите путь к файлу с эмбеддингами.");
            //    return;
            //}

            string filePath = "/Users/pablo_sergi/dockercerts/embeddings.pt"; //args[0];
            double[][] embeddings = LoadEmbeddings(filePath);

            int numClusters = 2;
            int[] clusterLabels = ClusterEmbeddings(embeddings, numClusters);

            Console.WriteLine("Метки кластеров:");
            for (int i = 0; i < clusterLabels.Length; i++)
            {
                Console.Write(clusterLabels[i] + " ");
            }
            Console.WriteLine();
        }
    }
}