package decisiontree;

import java.util.ArrayList;

public class AdaBooster {
    private ArrayList<DecisionTree>     m_arrTrees;
    private double[]                    m_arrAlphas;
    private double[]                    m_arrErrors;
    private ArrayList<Features_record>  m_arrTestDataSet;
    public AdaBooster(int x_nIterations)
    {
        m_arrTrees = new ArrayList<DecisionTree>();
        m_arrAlphas = new double[x_nIterations];
        m_arrErrors = new double[x_nIterations];
        for(int i = 0 ; i < x_nIterations ; i++)
        {
            DecisionTree t = new DecisionTree();
            t.SetMaxDepth(2);           
            m_arrTrees.add(t);
            m_arrAlphas[i] = 0;
        }
    }
    public void SetColumns(String[] x_arrCols)
    {
        for(DecisionTree t : m_arrTrees)
            t.CreateDataHeader(x_arrCols);
    }
    public void SetTrainingDataset(ArrayList<Features_record> x_ds)
    {
        m_arrTrees.get(0).SetDataset_train(x_ds);
    }
    public void SetTestDataset(ArrayList<Features_record> x_ds)
    {
        m_arrTestDataSet = x_ds;
    }
    public void RunAdaboost()
    {
        DecisionTree first_tree = m_arrTrees.get(0);
        first_tree.SetAllweightsEqual();
        first_tree.MakeTree();
        ArrayList<Features_record> updated_dataset = first_tree.GetUpdatedDataset();
        m_arrAlphas[0] = first_tree.GetAlpha();
        m_arrErrors[0] = first_tree.GetWeighted_error();
        
        for(int i = 1 ; i < m_arrTrees.size() ; i++)
        {
            DecisionTree tree = m_arrTrees.get(i);
            tree.SetDataset_train(updated_dataset);
            tree.MakeTree();
            updated_dataset = tree.GetUpdatedDataset();
            m_arrAlphas[i] = tree.GetAlpha();
            m_arrErrors[i] = tree.GetWeighted_error();
        }
    }
    public int Query(String[] x_arrFeatures)
    {
        int nClass = 0;
        double result = 0;
        for(int i = 0 ; i < m_arrTrees.size() ; i++)
        {
            DecisionTree t = m_arrTrees.get(i);
            result += m_arrAlphas[i] * t.Query(x_arrFeatures);
        }
        nClass = (result >= 0) ? 1 : -1;
        return nClass;
    }
    public double getAccuracyOnTestData()
    {
        double Accuracy = 0.0;
        for(Features_record rec : m_arrTestDataSet)
        {
            int nClass = (rec.GetLabel()) ? 1 : -1;
            if(Query(rec.GetAttributeArray()) == nClass)
                Accuracy++;
        }
        double acc_percentage = (Accuracy / m_arrTestDataSet.size())* 100.0;
        return acc_percentage;
    }
    public void printall()
    {
        for(DecisionTree t : m_arrTrees)
        {
            t.PrintTree();
            System.out.println("");
        }
    }
    public void print_errors()
    {
        for(int i = 0 ; i < m_arrTrees.size() ; i++)
        {
            System.out.printf("error iteration %d = %f\r\n", i,m_arrErrors[i]);
        }
    }
    
}
