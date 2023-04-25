using System;
using System.IO;
using TorchSharp;
using static TorchSharp.torch;
using Tensorflow;
using Accord.MachineLearning;
using Accord.Math.Distances;

partial class Program
{

    static void Main(string[] args)
    {
        if (args.Length == 0)
        {
            Console.WriteLine("Usage: TensorProcessor /Users/pablo_sergi/dockercerts/embeddings.pt");
            return;
        }

        string filePath = args[0];

        if (!File.Exists(filePath))
        {
            Console.WriteLine($"File '{filePath}' not found.");
            return;
        }

        double[][] inputTensors = LoadTensorFromFileAndConvertToDoubleArray(filePath);

        PrintDoubleArray(inputTensors);


        int numClusters = 2; // Укажите количество кластеров
        int[] clusterLabels = ClusterEmbeddings(inputTensors, numClusters);

        // Вывод меток кластеров на консоль
        Console.WriteLine("Метки кластеров:");
        for (int i = 0; i < clusterLabels.Length; i++)
        {
            Console.Write(clusterLabels[i] + " ");
        }
        Console.WriteLine();

    }

    static double[][] LoadTensorFromFileAndConvertToDoubleArray(string filePath)
    {
        var tensor = torch.Tensor.load(filePath);

        long[] shape = tensor.shape;
        long numRows = shape[0];
        long numCols = shape[1];

        double[][] result = new double[numRows][];

        for (int i = 0; i < numRows; i++)
        {
            result[i] = new double[numCols];

            for (int j = 0; j < numCols; j++)
            {
                result[i][j] = tensor[i, j].ToDouble();
            }
        }

        return result;
    }

    static void PrintDoubleArray(double[][] array)
    {
        for (int i = 0; i < array.Length; i++)
        {
            for (int j = 0; j < array[i].Length; j++)
            {
                Console.Write($"{array[i][j]} ");
            }
            Console.WriteLine();
        }
    }


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

}