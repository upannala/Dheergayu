pipeline{
    agent{
        label 'master'
    }
    
    stages{
        stage{
            steps{
                script{
                    git branch: param_map1.find{ it.key == "custom_branch" }.value, credentialsId:'', url: param_map1.find{ it.key == "custom_url" }.value
                    sh script: "cd custom && chmod +x custom.sh && ./custom.sh ${parameter_1} ${parameter_2}"
                    
                    def cmd= "aws ec2 describe-images --region eu-west-2 --filters Name=tag:Name,Values="+params.AMI+" --query Images[*].Tags[?Key==`MySqlRDSSnapID`].Value[] --output text"
                    
                        def command = $/ $cmd /$
                        def proc = command.execute()
                        proc.waitFor()              

                        def output = proc.in.text
                        def snap= output.tokenize()
                }
            }
        }
    }
}
