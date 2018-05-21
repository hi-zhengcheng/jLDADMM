import models.GibbsSamplingDMM;
import models.GibbsSamplingDMM_Inf;
import models.GibbsSamplingDMM_Inf_Single;
import models.GibbsSamplingLDA;
import models.GibbsSamplingLDA_Inf;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;

import utility.CmdArgs;
import eval.ClusteringEval;

/**
 * jLDADMM: A Java package for the LDA and DMM topic models
 * 
 * http://jldadmm.sourceforge.net/
 * 
 * @author: Dat Quoc Nguyen
 * 
 * @version: 1.0.2
 * 
 */
public class jLDADMM
{
	public static void main(String[] args)
	{
		CmdArgs cmdArgs = new CmdArgs();
		CmdLineParser parser = new CmdLineParser(cmdArgs);
		try {

			parser.parseArgument(args);

			if (cmdArgs.model.equals("LDA")) {
				GibbsSamplingLDA lda = new GibbsSamplingLDA(cmdArgs.corpus,
					cmdArgs.ntopics, cmdArgs.alpha, cmdArgs.beta,
					cmdArgs.niters, cmdArgs.twords, cmdArgs.expModelName,
					cmdArgs.initTopicAssgns, cmdArgs.savestep);
				lda.inference();
			}
			else if (cmdArgs.model.equals("DMM")) {
				GibbsSamplingDMM dmm = new GibbsSamplingDMM(cmdArgs.corpus,
					cmdArgs.ntopics, cmdArgs.alpha, cmdArgs.beta,
					cmdArgs.niters, cmdArgs.twords, cmdArgs.expModelName,
					cmdArgs.initTopicAssgns, cmdArgs.savestep);
				dmm.inference();
			}
			else if (cmdArgs.model.equals("LDAinf")) {
				GibbsSamplingLDA_Inf lda = new GibbsSamplingLDA_Inf(
					cmdArgs.paras, cmdArgs.corpus, cmdArgs.niters,
					cmdArgs.twords, cmdArgs.expModelName, cmdArgs.savestep);
				lda.inference();
			}
			else if (cmdArgs.model.equals("DMMinf")) {
				GibbsSamplingDMM_Inf dmm = new GibbsSamplingDMM_Inf(
					cmdArgs.paras, cmdArgs.corpus, cmdArgs.niters,
					cmdArgs.twords, cmdArgs.expModelName, cmdArgs.savestep);
				dmm.inference();
			}
			else if (cmdArgs.model.equals("DMM_Single")) {
				GibbsSamplingDMM_Inf_Single dmmSingle = new GibbsSamplingDMM_Inf_Single(
						cmdArgs.paras, cmdArgs.corpus, cmdArgs.niters,
						cmdArgs.twords, cmdArgs.expModelName, cmdArgs.savestep);
				dmmSingle.inferenceAllDoc();
			}
			else if (cmdArgs.model.equals("Eval")) {
				ClusteringEval.evaluate(cmdArgs.labelFile, cmdArgs.dir,
					cmdArgs.prob);
			}
			else {
				System.out
					.println("Error: Option \"-model\" must get \"LDA\" or \"DMM\" or \"LDAinf\" or \"DMMinf\" or \"Eval\"");
				System.out
					.println("\tLDA: Specify the Latent Dirichlet Allocation topic model");
				System.out
					.println("\tDMM: Specify the one-topic-per-document Dirichlet Multinomial Mixture model");
				System.out
					.println("\tLDAinf: Infer topics for unseen corpus using a pre-trained LDA model");
				System.out
					.println("\tDMMinf: Infer topics for unseen corpus using a pre-trained DMM model");
				System.out
					.println("\tEval: Specify the document clustering evaluation");
				help(parser);
				return;
			}
		}
		catch (CmdLineException cle) {
			System.out.println("Error: " + cle.getMessage());
			help(parser);
			return;
		}
		catch (Exception e) {
			System.out.println("Error: " + e.getMessage());
			e.printStackTrace();
			return;
		}
	}

	public static void help(CmdLineParser parser)
	{
		System.out
			.println("java -jar jLDADMM.jar [options ...] [arguments...]");
		parser.printUsage(System.out);
	}
}
